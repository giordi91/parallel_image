#include <ui/float_widget.h>



FloatWidget::FloatWidget(QWidget *par, Attribute * attr,
						MainWindow * main_widget, Filter * f)
    : QWidget(par)
{
	ui.setupUi(this);
	m_attribute = static_cast<AttributeTyped<float>* >(attr);
	m_main_widget = main_widget;
	m_f = f;

	connect(ui.valueDSB, SIGNAL(valueChanged(double)), this, SLOT(set_value(double)));
	connect(ui.valueSL, SIGNAL(valueChanged(int)), this, SLOT(set_value(int)));


	//connect attributes

}

void FloatWidget::set_value(double value)
{
	m_attribute->set_value((float)value);
	ui.valueSL->setValue((int)value);
	m_f->update_data();
	m_main_widget->update_image();
}

void FloatWidget::set_value(int value)
{
	// int value = ui.valueSL.value();
	m_attribute->set_value((float)value);
	ui.valueDSB->setValue((double)value);
	m_f->update_data();
	m_main_widget->update_image();
}