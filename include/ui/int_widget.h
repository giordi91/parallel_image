#include "ui_int_widget.h"
#include <QtWidgets/QWidget>
#include <QtCore/QString>
#include <string>

#ifndef __PARALLEL_INT_WIDGET_H__
#define __PARALLEL_INT_WIDGET_H__


class IntWidget : public QWidget
{
	Q_OBJECT

public:
	explicit IntWidget(QWidget *parent = 0);

	// void set_filter_name(std::string name);

private:
	Ui_int_widget ui;
public slots:
	
};

#endif