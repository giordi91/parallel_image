#include <core/filter_manager.h>

Filter_manager::Filter_manager(Bitmap * bmp):m_bmp(bmp)
{
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