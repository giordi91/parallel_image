/**
 * @brief this is a abstract stancil interface
 * 
 * This class is used to implement an abstract implementation for 
 * convolution kernels so to do many differents effects by just changing the 
 * kind of stancil we wants to apply
 */

#ifndef __PARALLEL_IMAGE_STANCIL_H__
#define __PARALLEL_IMAGE_STANCIL_H__
#include <cstddef> 
#include <cstdint> 

 class Stancil
 {
 public:
 	/**
 	 * @brief returns the stancil value at wanted coordinates
 	 * 		  
 	 * @details this function implents [] used to access the internal
 	 * 			data of the generated stancil 
 	 * @param x: the x coordinate of the stancil
 	 * @param y: the y coordinate of the stancil
 	 */
 	virtual float get_value(size_t x, size_t y)const = 0;
 	
 	/**
 	 * @brief printing at conosle the stancil values
 	 * @details this is a utility function printing at console the value
 	 * 			generated in the stancil
 	 */
 	virtual void log()const = 0;
 	
 	/**
 	 * @brief get the width of the stancil
 	 * @return width in pixel of the stancil
 	 */
 	virtual size_t get_width()const = 0;  
 	
 	/**
 	 * @brief get the height of the stancil
 	 * @return height in pixel of the stancil
 	 */
 	virtual size_t get_height()const = 0;
	
	/**
 	 * @brief the they stancil size in byte
 	 * @details returns the stancil size in bytes , mostly used for
 	 * 			memory copy purpose, for example transfering the stancil on 
 	 * 			the GPU
 	 * @return the byte size of the stancil
 	 */
 	virtual size_t get_size()const = 0;  

 	/**
 	 * @brief get pointer to the stancil data
 	 * @details returns a read only pointer to the stancil data
 	 */
 	virtual const float * get_data()const = 0 ;

 	/**
 	 * @brief the stancil destuctor
 	 */
 	virtual ~Stancil(){}
 	
 };

/**
 * @brief generic stancil implementation
 * @details this is a generic stancil impelemntation which can be used for 
 * 			generic operation like sharpening or edge detection
 */	
class GenericStancil: public Stancil
 {
 public:
 	/**
 	 * @brief this is the constructor
 	 * @details this is a generic constructor that makes
 	 * 			and internal copy of a given array
 	 * @param data: a pointer to an array of floats defining the stancil
 	 * @param width: the width of the stancil
 	 * @param height: the height of the stancil
 	 */
 	GenericStancil();
 	GenericStancil(	const float * data, 
 					const size_t width, 
					const size_t height);
 	/**
 	 * @brief returns the stancil value at wanted coordinates
 	 * 		  
 	 * @details this function implents [] used to access the internal
 	 * 			data of the generated stancil 
 	 * @param x: the x coordinate of the stancil
 	 * @param y: the y coordinate of the stancil
 	 */
 	inline float get_value(size_t x, size_t y)const ;
 	
 	/**
 	 * @brief printing at conosle the stancil values
 	 * @details this is a utility function printing at console the value
 	 * 			generated in the stancil
 	 */
 	void log()const ;
 	
 	/**
 	 * @brief get the width of the stancil
 	 * @return width in pixel of the stancil
 	 */
 	size_t get_width()const ;  
 	
 	/**
 	 * @brief get the height of the stancil
 	 * @return height in pixel of the stancil
 	 */
 	size_t get_height()const ;
	
	/**
 	 * @brief the they stancil size in byte
 	 * @details returns the stancil size in bytes , mostly used for
 	 * 			memory copy purpose, for example transfering the stancil on 
 	 * 			the GPU
 	 * @return the byte size of the stancil
 	 */
 	size_t get_size()const ;  

 	/**
 	 * @brief get pointer to the stancil data
 	 * @details returns a read only pointer to the stancil data
 	 */
 	const float * get_data()const  ;

 	virtual ~GenericStancil();


 protected:
 	float * m_data;
 	size_t m_width;
 	size_t m_height;
 };


 #endif

