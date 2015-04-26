#include <ui/filter_widget.h>


FilterWidget::FilterWidget(QWidget *par)
    : QWidget(par)
{
	ui.setupUi(this);
}

void FilterWidget::set_filter_name(std::string name)
{
	ui.mainGB->setTitle(name.c_str());
}