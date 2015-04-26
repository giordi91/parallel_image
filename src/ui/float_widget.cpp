#include <ui/float_widget.h>



FloatWidget::FloatWidget(QWidget *par, Attribute * attr)
    : QWidget(par)
{
	ui.setupUi(this);
	m_attribute = static_cast<AttributeTyped<float>* >(attr);

	//connect attributes

}

void FloatWidget::set_value(double value)
{
	m_attribute->set_value((float)value);
	ui.valueSL->setValue((int)value);

}

void FloatWidget::set_value(int value)
{
	m_attribute->set_value((float)value);
	ui.valueDSB->setValue((double)value);
}