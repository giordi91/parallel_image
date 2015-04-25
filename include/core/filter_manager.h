#include <cstddef> 
#include <cstdint> 
#include <vector>
#include <string>

#include <core/filter.h>
#include <core/bitmap.h>
#include <core/GPU_manager.h>
#include <cstdint> // uint8_t declaration
#include <unordered_map>


using std::vector;
using std::string;

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
	/**
	 * @brief set what type of computation to use
	 * @param type: Computation, the wanted computation
	 */
	void set_compute_type(const Computation type);

	/**
	 * @brief returns the currenct computation type
	 * @return Computation
	 */
	Computation get_compute_type() const;

	/**
	 * @brief this function computes the stack of filters
	 * @details This function kicks out the evaluation stack
	 * of the filters, the computation type is defined by 
	 * the internal valirable  m_comp_type which is of type
	 * Computation and can be set with the set_compute_type
	 * function, by default is set to SERIAL
	 */
	void evaluate_stack();

	/**
	 * @brief Makes an internal copy of the image input
	 * @details This function makes a copy of the input buffer
	 * in order to avoid to override the input data. This is 
	 * done because internally the stack is evaluated by swapping
	 * pointers of target and source between each filter in order
	 * to avoid useless memcpy and to optimize the output.
	 */
	void copy_input_buffer();

	/**
	 * @brief saves the result of evaluation
	 * @param path Where we want to save the image
	 * @throw : runtime_error if path does not exists
	 */
	void save_stack_output(const char* path);

	/**
	 * @brief Add a filters using the wanted name
	 * @details This function act as a factory, it takes as input
	 * 			a name of the wanted filter to istantiate, and if
	 * 			the name is in the map it makes an istances of it and
	 * 			adds it to the internal filters array, if not throws
	 * 
	 * @param name the name of the filters to add
	 * @thorw invalid_argument
	 */
	void add_filter_by_name(const char * name);


	static vector<string> get_available_filters_name();



private:
	/**
	 * @brief swapping the buffer between one filter and the other
	 	
	 * @details [long description]
	 */
	void swap_buffers(size_t current_index,
					  size_t final_index);
	/**
	 * @brief This function is in charge of swapping the buffers
	 * @details The swapping buffers operation is used between 
	 *          each filter evaluation, rather then copying the 
	 *          data we just swap the pointers, and target becomes
	 *          source and viceversa
	 */
	void setup_buffers();

private:
	//the internal filters data (pointers to the actual filters)
	vector<Filter *>m_filters;
	//the pointer to the image to work on
	Bitmap * m_bmp;
	//the computation state of the manager
	Computation m_comp_type;
	//output bmp
	Bitmap *m_out_bmp;
	//a copy of the original input image
	uint8_t * m_input_copy;
	//stack start variable
	size_t m_stack_start;
	//Temporary buffer pointer , used for the swap
	uint8_t * working_buffer;
	//The source buffer pointer
	uint8_t * source_buffer;
	//The target buffer pointer
	uint8_t * target_buffer;

	//Pointer to the GPU manager 
	GPU_manager * m_gpu_manager;

	//Typedef for our function pointer
	//We use this to point to the static creation classes
	//of the filters
	typedef Filter* (*FilterFunc)(const int&, const int&);
	//typedef for the map holding pointers to the crate functions
	typedef std::unordered_map<std::string, FilterFunc> function_map;
	//the static map of function pointers
	static function_map m_functions;


};

#endif

