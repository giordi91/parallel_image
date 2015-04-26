#include <ui/int_widget.h>


IntWidget::IntWidget(QWidget *par, Attribute* attr)
    : QWidget(par)
{
	ui.setupUi(this);
	m_attribute = static_cast<AttributeTyped<int>* >(attr);

}

// void IntWidget::set_value()
// {}