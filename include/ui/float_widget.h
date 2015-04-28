#include <ui/ui_float_widget.h>
#include <QtWidgets/QWidget>
#include <QtCore/QString>
#include <string>
#include <core/attribute.h>
#include <ui/mainwindow.h>
#include <core/filter.h>

#ifndef __PARALLEL_FLOAT_WIDGET_H__
#define __PARALLEL_FLOAT_WIDGET_H__


class FloatWidget : public QWidget
{
	Q_OBJECT

public:
	explicit FloatWidget(QWidget *parent, Attribute * attr, 
						MainWindow * main_widget, Filter * f);

	Ui_float_widget ui;
private:
	AttributeTyped<float> * m_attribute;
	MainWindow * m_main_widget;
	Filter * m_f;
signals:

public slots:
	void set_value(double value);
	void set_value(int value);
	
};

#endif