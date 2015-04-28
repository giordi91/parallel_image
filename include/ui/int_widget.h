#include "ui_int_widget.h"
#include <QtWidgets/QWidget>
#include <QtCore/QString>
#include <string>
#include <core/attribute.h>
#include <ui/mainwindow.h>
#include <core/filter.h>


#ifndef __PARALLEL_INT_WIDGET_H__
#define __PARALLEL_INT_WIDGET_H__


class IntWidget : public QWidget
{
	Q_OBJECT

public:
	explicit IntWidget(QWidget *parent, Attribute * attr, 
						MainWindow * main_widget, Filter * f);


	Ui_int_widget ui;
private:
	AttributeTyped<int> * m_attribute;
	MainWindow * m_main_widget;
	Filter * m_f;
private slots:
	void update(int i);


};

#endif