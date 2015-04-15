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

		AttributeTyped(const char *name):m_name(name){};
		AttributeTyped(const char *name ,T value):m_name(name),m_value(value){};

		std::string type()
		{
			return typeid(T).name();
		}

		T get_value() const {return m_value;};
		void set_value(T value){ m_value=value;};
		std::string get_name(){return m_name;};
	private:

		std::string m_name;
		T m_value;
};


#endif
