#include <cstddef> 
#include <cstdint> 
#include <vector>

#include <core/filter.h>
#include <core/bitmap.h>

using std::vector;

#ifndef __PARALLEL_IMAGE_FILTER_MANAGER_H__
#define __PARALLEL_IMAGE_FILTER_MANAGER_H__


class Filter_manager
{
public:
	/**
	 * @brief the constructor
	 */
	Filter_manager(Bitmap * bmp);
	~Filter_manager();
	/**
	 * @brief add a filter to the manager
	 * @details This function allows to add a filter to the internal 
	 * datastructore 
	 * 
	 * @param fil a poibter the newly created filter, most likely generated from
	 * the factory class
	 */
	void add_filter(Filter * fil);

private:
	vector<Filter *>m_filters;
	Bitmap * m_bmp;



};

#endif