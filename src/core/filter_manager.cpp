#include <core/filter_manager.h>
#include <stdexcept>      // std::invalid_argument
#include <iostream>
#include <cstring> // memcpy
#include <tbb/task_scheduler_init.h>

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

}

Filter_manager::~Filter_manager()
{
	if (m_bmp)
	{
		delete m_bmp;
	}
	for (auto fil : m_filters)
	{
		delete fil;
	}

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
	source_buffer = m_input_copy;
	target_buffer = m_out_bmp->getRawData();

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

	m_out_bmp->save("/home/giordi/WORK_IN_PROGRESS/C/parallel_image/data/jessyBW.bmp");
}


void Filter_manager::copy_input_buffer()
{
    size_t buffer_size = m_bmp->get_width()*m_bmp->get_height()*3*(size_t)sizeof(uint8_t);
	m_input_copy = new uint8_t[buffer_size];
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
		target_buffer = m_out_bmp->getRawData();
	}
}