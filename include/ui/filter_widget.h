#include "ui_filter_widget.h"
#include <QtWidgets/QWidget>
#include <core/filter_manager.h>
#include <QtCore/QString>
#include <string>

#ifndef __PARALLEL_FILTER_WIDGET_H__
#define __PARALLEL_FILTER_WIDGET_H__


class FilterWidget : public QWidget
{
	Q_OBJECT

public:
	explicit FilterWidget(QWidget *parent = 0);

	void set_filter_name(std::string name);

private:
	Ui_Filter_Widget ui;
public slots:
	
};

#endif