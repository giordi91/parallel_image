#include <gmock/gmock.h>
#include <iostream>
#include <core/attribute.h>
using namespace testing;



TEST(Attribute_test, basic_instances)
{
	AttributeTyped<int> intAttr("intTest");
	AttributeTyped<float> floatAttr("floatTest");
	AttributeTyped<bool> boolAttr("boolTest");
}

TEST(Attribute_test, default_value)
{
	AttributeTyped<int> intAttr("intTest",11);
	AttributeTyped<float> floatAttr("floatTest",1.2f);
	AttributeTyped<bool> boolAttr("boolTest",true);

	EXPECT_EQ(intAttr.get_value(),11);
	EXPECT_EQ(floatAttr.get_value(),1.2f);
	EXPECT_EQ(boolAttr.get_value(),true);

}

TEST(Attribute_test, set_get_value)
{
	AttributeTyped<int> intAttr("intTest");
	AttributeTyped<float> floatAttr("floatTest");
	AttributeTyped<bool> boolAttr("boolTest");

	intAttr.set_value(10);	
	floatAttr.set_value(5.0f);
	boolAttr.set_value(false);	

	EXPECT_EQ(intAttr.get_value(),10);
	EXPECT_EQ(floatAttr.get_value(),5.0f);
	EXPECT_EQ(boolAttr.get_value(),false);


	intAttr.set_value(-1);	
	floatAttr.set_value(15.2f);
	boolAttr.set_value(true);	

	EXPECT_EQ(intAttr.get_value(),-1);
	EXPECT_EQ(floatAttr.get_value(),15.2f);
	EXPECT_EQ(boolAttr.get_value(),true);
}

TEST(Attribute_test, ensure_name)
{
	AttributeTyped<int> intAttr("intTest");
	AttributeTyped<float> floatAttr("floatTest");
	AttributeTyped<bool> boolAttr("boolTest");

	EXPECT_EQ(intAttr.get_name(),"intTest");
	EXPECT_EQ(floatAttr.get_name(),"floatTest");
	EXPECT_EQ(boolAttr.get_name(),"boolTest");
}

//specs
//template attribute
//get set value
//get_set name