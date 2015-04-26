#include <ui/ui_float_widget.h>
#include <QtWidgets/QWidget>
#include <QtCore/QString>
#include <string>

#ifndef __PARALLEL_FLOAT_WIDGET_H__
#define __PARALLEL_FLOAT_WIDGET_H__


class FloatWidget : public QWidget
{
	Q_OBJECT

public:
	explicit FloatWidget(QWidget *parent = 0);

	// void set_filter_name(std::string name);

private:
	Ui_float_widget ui;
public slots:
	
};

#endif