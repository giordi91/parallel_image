#include <core/filter_manager.h>

Filter_manager::Filter_manager(Bitmap * bmp):m_bmp(bmp)
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

size_t Filter_manager::stack_size()
{	
	return m_filters.size();
}