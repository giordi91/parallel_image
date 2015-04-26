#include "ui_filter_widget.h"
#include <QtWidgets/QWidget>
#include <core/filter_manager.h>
#include <QtCore/QString>

#ifndef __PARALLEL_FILTER_WIDGET_H__
#define __PARALLEL_FILTER_WIDGET_H__


class FilterWidget : public QWidget
{
	Q_OBJECT

public:
	explicit FilterWidget(QWidget *parent = 0);

private:
	Ui_Filter_Widget ui;
public slots:
	
};

#endif