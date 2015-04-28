#include <core/filter_manager.h>
#include <stdexcept>      // std::invalid_argument
#include <iostream>
#include <cstring> // memcpy
#include <tbb/task_scheduler_init.h>
#include <filters/bw_filter.h>
#include <filters/edge_detection_filter.h>
#include <filters/gaussian_filter.h>
#include <filters/sharpen_filter.h>

//static map initialization
Filter_manager::function_map Filter_manager::m_functions= {
	{"Bw_filter",&Bw_filter::create_filter},	
	{"Edge_detection_filter",&Edge_detection_filter::create_filter},	
	{"Gaussian_filter",&Gaussian_filter::create_filter},	
	{"Sharpen_filter",&Sharpen_filter::create_filter},	
};

Filter_manager::Filter_manager(Bitmap * bmp):m_bmp(bmp),
										     m_comp_type(SERIAL),
										     m_out_bmp(nullptr),
										     m_input_copy(nullptr),
										     m_stack_start(0),
										     working_buffer(nullptr)
{
	m_filters.resize(0);

	//allocating output bmp
	unsigned int width = m_bmp->get_width();
    unsigned int height = m_bmp->get_height();
    unsigned int padded_size = m_bmp->get_padded_size();
    m_out_bmp = new Bitmap(width, height, padded_size);

    // //lets make a copy of the original input
    copy_input_buffer();

    //initializing TBB
    tbb::task_scheduler_init init;

    //initialize gpu manager
    m_gpu_manager = new GPU_manager(m_bmp->get_width(),
    								m_bmp->get_height(),
    								16);

    m_gpu_manager->copy_data_to_device(m_input_copy, 
    									m_gpu_manager->get_source_buffer() );	
}

Filter_manager::~Filter_manager()
{
	// if (m_bmp)
	// {
	// 	delete m_bmp;
	// }
	// for (auto fil : m_filters)
	// {
	// 	delete fil;
	// }

	m_filters.clear();

	//delete the output_bmp
	if (m_out_bmp)
	{
		delete m_out_bmp;
	}
	if (m_input_copy)
	{
		delete m_input_copy;
	}

	if (m_gpu_manager)
	{
		delete m_gpu_manager;
	}
}

void Filter_manager::add_filter(Filter * fil)
{
	m_filters.push_back(fil);
}

size_t Filter_manager::stack_size() const
{	
	return m_filters.size();
}

Filter * Filter_manager::operator[](size_t value)
{
	//should i check for boundaries?
	return m_filters[value];
}


void Filter_manager::remove_filter(const size_t index)
{
	if (index < stack_size())
	{
		delete m_filters[index];
		m_filters.erase( m_filters.begin()+ index);
	}
	else
	{
		throw std::invalid_argument("Filter_manager::remove_filter : index out of range");
	}
}


Filter * Filter_manager::pop_filter(const size_t index)
{

	if (index < stack_size())
	{
		auto temp = m_filters[index];
		m_filters.erase( m_filters.begin()+ index);
		return temp;
	}
	else
	{
		throw std::invalid_argument("Filter_manager::remove_filter : index out of range");
	}

}


void Filter_manager::set_compute_type(const Computation type)
{
	m_comp_type = type;
}

Filter_manager::Computation Filter_manager::get_compute_type() const
{
	return m_comp_type;
}


void Filter_manager::evaluate_stack()
{
	size_t st_size = stack_size();
	setup_buffers();

	for (size_t i=m_stack_start; i<st_size; ++i)
	{
		if(m_comp_type == SERIAL)
		{
			m_filters[i]->compute_serial(source_buffer,
										 target_buffer);
		}
		else if(m_comp_type == TBB)
		{
			m_filters[i]->compute_tbb(source_buffer,
									  target_buffer);
		}
		else if(m_comp_type == CUDA)
		{
			m_filters[i]->compute_cuda(source_buffer,
									    target_buffer);
		}

		swap_buffers(i,st_size);
	}

	//if data was on the GPU copy it back, not sure if this is needed,
	//probably only needed when we want to save the image, let see
	//when we implement the UI
	if (m_comp_type == CUDA)
	{
		m_gpu_manager->copy_data_from_device( source_buffer,
										 m_out_bmp->getRawData());
	}
}


void Filter_manager::copy_input_buffer()
{
    size_t buffer_size = m_bmp->get_width()*m_bmp->get_height()*3*(size_t)sizeof(uint8_t);
    if (!m_input_copy)
    {
		m_input_copy = new uint8_t[buffer_size];
    }

	memcpy(m_input_copy, m_bmp->getRawData(), buffer_size);
}


void Filter_manager::swap_buffers(size_t current_index,
					  				size_t final_index)
{

	working_buffer = target_buffer;
	target_buffer = source_buffer;
	source_buffer = working_buffer;

	if (current_index == (final_index-2))
	{
		if (m_comp_type == SERIAL || m_comp_type == TBB)
		{
			target_buffer = m_out_bmp->getRawData();
		}

	}
}

void Filter_manager::setup_buffers()
{
	//copying a fresh copy of the input
	copy_input_buffer();
	if (m_comp_type == SERIAL || m_comp_type == TBB)
	{
		source_buffer = m_input_copy;
		target_buffer = m_out_bmp->getRawData();
	}
	else if (m_comp_type == CUDA)
	{
	    //if cuda let s do a fresh copy on the gpu aswell
	    m_gpu_manager->copy_data_to_device(m_input_copy, 
									m_gpu_manager->get_source_buffer() );	
		//setup the buffers for the gpu
		source_buffer = m_gpu_manager->get_source_buffer();
		target_buffer = m_gpu_manager->get_target_buffer();
	}
}

void Filter_manager::save_stack_output(const char* path)
{
	m_out_bmp->save(path);
}


void Filter_manager::add_filter_by_name(const char *name)
{
	if (m_functions.count(name) > 0)
	{
		Filter * f = m_functions[name](m_bmp->get_width(),
									m_bmp->get_height());
		add_filter(f);
	}
	else
	{
		throw std::invalid_argument("Filter type is not part of the map");
	}
}


vector<string> Filter_manager::get_available_filters_name()
{
	vector<string> vec;
	for (auto iter : m_functions)
	{
		vec.push_back(iter.first);
	}
	return vec;
}


vector<Filter *> Filter_manager::get_filters_data()const
{
	return m_filters;
}


uint8_t * Filter_manager::get_output_buffer()
{
	return m_out_bmp->getRawData();
}