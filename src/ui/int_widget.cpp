#include <ui/int_widget.h>


IntWidget::IntWidget(QWidget *par, Attribute* attr,
	MainWindow * main_widget, Filter * f)
    : QWidget(par)
{
	ui.setupUi(this);
	m_attribute = static_cast<AttributeTyped<int>* >(attr);
	m_main_widget = main_widget;
	m_f = f;

	connect(ui.valueSB, SIGNAL(valueChanged(int)), this, SLOT(update(int)));


}

void IntWidget::update(int i)
{
	m_attribute->set_value((size_t)i);
	m_f->update_data();
	m_main_widget->update_image();
}