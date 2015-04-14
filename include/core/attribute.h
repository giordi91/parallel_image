#include <string>
#include <typeinfo>

#ifndef __PARALLEL_IMAGE_ATTRIBUTE_H__
#define __PARALLEL_IMAGE_ATTRIBUTE_H__ 

class Attribute {
	public:
		virtual ~Attribute() = 0;

		virtual std::string type() = 0;	

};

template<typename T>
class AttributeTyped {
	public:

		
		std::string type()
		{
			return typeid(T).name();
		}

	private:

		T m_value;
};


#endif
