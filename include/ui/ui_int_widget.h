/********************************************************************************
** Form generated from reading UI file 'int_widget.ui'
**
** Created by: Qt User Interface Compiler version 5.4.0
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_INT_WIDGET_H
#define UI_INT_WIDGET_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QSpinBox>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_int_widget
{
public:
    QHBoxLayout *horizontalLayout;
    QLabel *attributeL;
    QSpinBox *valueSB;

    void setupUi(QWidget *int_widget)
    {
        if (int_widget->objectName().isEmpty())
            int_widget->setObjectName(QStringLiteral("int_widget"));
        int_widget->resize(400, 49);
        horizontalLayout = new QHBoxLayout(int_widget);
        horizontalLayout->setSpacing(3);
        horizontalLayout->setContentsMargins(11, 11, 11, 11);
        horizontalLayout->setObjectName(QStringLiteral("horizontalLayout"));
        horizontalLayout->setContentsMargins(3, 3, 3, 3);
        attributeL = new QLabel(int_widget);
        attributeL->setObjectName(QStringLiteral("attributeL"));
        QSizePolicy sizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(attributeL->sizePolicy().hasHeightForWidth());
        attributeL->setSizePolicy(sizePolicy);

        horizontalLayout->addWidget(attributeL);

        valueSB = new QSpinBox(int_widget);
        valueSB->setObjectName(QStringLiteral("valueSB"));

        horizontalLayout->addWidget(valueSB);


        retranslateUi(int_widget);

        QMetaObject::connectSlotsByName(int_widget);
    } // setupUi

    void retranslateUi(QWidget *int_widget)
    {
        int_widget->setWindowTitle(QApplication::translate("int_widget", "Widget", 0));
        attributeL->setText(QApplication::translate("int_widget", "attribute", 0));
    } // retranslateUi

};

namespace Ui {
    class int_widget: public Ui_int_widget {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_INT_WIDGET_H
