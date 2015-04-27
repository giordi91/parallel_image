#include <ui/ui_float_widget.h>
#include <QtWidgets/QWidget>
#include <QtCore/QString>
#include <string>
#include <core/attribute.h>

#ifndef __PARALLEL_FLOAT_WIDGET_H__
#define __PARALLEL_FLOAT_WIDGET_H__


class FloatWidget : public QWidget
{
	Q_OBJECT

public:
	explicit FloatWidget(QWidget *parent = 0, Attribute * attr=nullptr);

	Ui_float_widget ui;
private:
	AttributeTyped<float> * m_attribute;
signals:

public slots:
	void set_value(double value);
	void set_value(int value);
	
};

#endif