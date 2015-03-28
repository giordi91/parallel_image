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
	enum Computation { SERIAL=0, TBB=1, CUDA=2 };

	/**
	 * @brief the constructor
	 */
	Filter_manager(Bitmap * bmp);
	/**
	 * @brief this is the destructor
	 * @details The destructor is in charge of deleting 
	 * all the stuff the manager took ownership of, like 
	 * the filters and the input bitmap,this mean there wont
	 * be need of housekeeping outside the File_manager
	 */
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
	/**
	 * @brief returns the size of the filter stack
	 * @return size_t, the size
	 */
	size_t stack_size() const;
	/**
	 * @brief subscription operator
	 * @details the subscription operator gives you a pointer to the
	 *  wanted filter
	 * 
	 * @param value the index to access
	 */
	Filter * operator[](size_t value);

	/**
	 * @brief removing a filter from manager
	 * @details this function deletes the filters at current
	 * index
	 * 
	 * @param index the index we want to delete
	 * @thow: invalid_argument if out of range
	 */
	void remove_filter(const size_t index);

	/**
	 * @brief pop out a filter from manager
	 * @details this function removes  the filters at current
	 * index without deleteing it and returns the wanted pointer,
	 * from that moment the manager has no more ownership over that 
	 * memory is up to the user to clean up
	 * 
	 * @param index the index we want to pop
	 * @thow: invalid_argument if out of range
	 */
	Filter * pop_filter(const size_t index);

	void set_compute_type(const Computation type);
	Computation get_compute_type() const;

private:
	//the internal filters data (pointers to the actual filters)
	vector<Filter *>m_filters;
	//the pointer to the image to work on
	Bitmap * m_bmp;
	//the computation state of the manager
	Computation m_comp_type;

};

#endif