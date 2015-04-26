/********************************************************************************
** Form generated from reading UI file 'float_widget.ui'
**
** Created by: Qt User Interface Compiler version 5.4.0
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_FLOAT_WIDGET_H
#define UI_FLOAT_WIDGET_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QDoubleSpinBox>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QSlider>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_float_widget
{
public:
    QHBoxLayout *horizontalLayout;
    QLabel *attributeL;
    QSlider *valueSL;
    QDoubleSpinBox *valueDSB;

    void setupUi(QWidget *float_widget)
    {
        if (float_widget->objectName().isEmpty())
            float_widget->setObjectName(QStringLiteral("float_widget"));
        float_widget->resize(400, 49);
        horizontalLayout = new QHBoxLayout(float_widget);
        horizontalLayout->setSpacing(3);
        horizontalLayout->setContentsMargins(11, 11, 11, 11);
        horizontalLayout->setObjectName(QStringLiteral("horizontalLayout"));
        horizontalLayout->setContentsMargins(3, 3, 3, 3);
        attributeL = new QLabel(float_widget);
        attributeL->setObjectName(QStringLiteral("attributeL"));
        QSizePolicy sizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(attributeL->sizePolicy().hasHeightForWidth());
        attributeL->setSizePolicy(sizePolicy);

        horizontalLayout->addWidget(attributeL);

        valueSL = new QSlider(float_widget);
        valueSL->setObjectName(QStringLiteral("valueSL"));
        valueSL->setValue(1);
        valueSL->setOrientation(Qt::Horizontal);

        horizontalLayout->addWidget(valueSL);

        valueDSB = new QDoubleSpinBox(float_widget);
        valueDSB->setObjectName(QStringLiteral("valueDSB"));
        valueDSB->setValue(1);

        horizontalLayout->addWidget(valueDSB);


        retranslateUi(float_widget);

        QMetaObject::connectSlotsByName(float_widget);
    } // setupUi

    void retranslateUi(QWidget *float_widget)
    {
        float_widget->setWindowTitle(QApplication::translate("float_widget", "Widget", 0));
        attributeL->setText(QApplication::translate("float_widget", "attribute", 0));
    } // retranslateUi

};

namespace Ui {
    class float_widget: public Ui_float_widget {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_FLOAT_WIDGET_H
