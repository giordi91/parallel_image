#include <cstdint> // uint8_t declaration
#include <gmock/gmock.h>
#include <iostream>
#include <stdexcept>      // std::invalid_argument
#include <core/attribute.h>
using namespace testing;



TEST(Attribute_test, basic_instances)
{
	AttributeTyped<int> intAttr;
	AttributeTyped<float> floatAttr;

	std::cout<<intAttr.type()<<std::endl;
	std::cout<<floatAttr.type()<<std::endl;

}