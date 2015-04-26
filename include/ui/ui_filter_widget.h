/********************************************************************************
** Form generated from reading UI file 'filter_widget.ui'
**
** Created by: Qt User Interface Compiler version 5.4.0
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_FILTER_WIDGET_H
#define UI_FILTER_WIDGET_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QCheckBox>
#include <QtWidgets/QGroupBox>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_Filter_Widget
{
public:
    QVBoxLayout *verticalLayout;
    QGroupBox *mainGB;
    QVBoxLayout *verticalLayout_2;
    QCheckBox *evalCB;

    void setupUi(QWidget *Filter_Widget)
    {
        if (Filter_Widget->objectName().isEmpty())
            Filter_Widget->setObjectName(QStringLiteral("Filter_Widget"));
        Filter_Widget->resize(400, 80);
        verticalLayout = new QVBoxLayout(Filter_Widget);
        verticalLayout->setSpacing(6);
        verticalLayout->setContentsMargins(11, 11, 11, 11);
        verticalLayout->setObjectName(QStringLiteral("verticalLayout"));
        mainGB = new QGroupBox(Filter_Widget);
        mainGB->setObjectName(QStringLiteral("mainGB"));
        verticalLayout_2 = new QVBoxLayout(mainGB);
        verticalLayout_2->setSpacing(6);
        verticalLayout_2->setContentsMargins(11, 11, 11, 11);
        verticalLayout_2->setObjectName(QStringLiteral("verticalLayout_2"));
        verticalLayout_2->setContentsMargins(-1, 20, -1, -1);
        evalCB = new QCheckBox(mainGB);
        evalCB->setObjectName(QStringLiteral("evalCB"));

        verticalLayout_2->addWidget(evalCB);


        verticalLayout->addWidget(mainGB);


        retranslateUi(Filter_Widget);

        QMetaObject::connectSlotsByName(Filter_Widget);
    } // setupUi

    void retranslateUi(QWidget *Filter_Widget)
    {
        Filter_Widget->setWindowTitle(QApplication::translate("Filter_Widget", "Widget", 0));
        mainGB->setTitle(QApplication::translate("Filter_Widget", "FILTER NAME", 0));
        evalCB->setText(QApplication::translate("Filter_Widget", "evaluate", 0));
    } // retranslateUi

};

namespace Ui {
    class Filter_Widget: public Ui_Filter_Widget {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_FILTER_WIDGET_H
