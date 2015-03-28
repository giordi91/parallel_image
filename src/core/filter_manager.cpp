#include <core/filter_manager.h>
#include <stdexcept>      // std::invalid_argument

Filter_manager::Filter_manager(Bitmap * bmp):m_bmp(bmp),
										     m_comp_type(SERIAL)
{
	m_filters.resize(0);
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