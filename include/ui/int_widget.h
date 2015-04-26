#include "ui_int_widget.h"
#include <QtWidgets/QWidget>
#include <QtCore/QString>
#include <string>
#include <core/attribute.h>

#ifndef __PARALLEL_INT_WIDGET_H__
#define __PARALLEL_INT_WIDGET_H__


class IntWidget : public QWidget
{
	Q_OBJECT

public:
	explicit IntWidget(QWidget *parent = 0, Attribute * attr=nullptr);


private:
	Ui_int_widget ui;
	AttributeTyped<int> * m_attribute;
public slots:

};

#endif