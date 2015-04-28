/********************************************************************************
** Form generated from reading UI file 'base_window.ui'
**
** Created by: Qt User Interface Compiler version 5.4.0
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_BASE_WINDOW_H
#define UI_BASE_WINDOW_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QCheckBox>
#include <QtWidgets/QComboBox>
#include <QtWidgets/QGroupBox>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenu>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QScrollArea>
#include <QtWidgets/QSplitter>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>
#include "ui/texturewidget.h"

QT_BEGIN_NAMESPACE

class Ui_MainWindow
{
public:
    QAction *actionOpen;
    QAction *actionSave;
    QWidget *centralWidget;
    QVBoxLayout *verticalLayout_3;
    QWidget *widget;
    QHBoxLayout *horizontalLayout_4;
    QGroupBox *compGB;
    QHBoxLayout *horizontalLayout_2;
    QLabel *compL;
    QComboBox *compCB;
    QGroupBox *filterGB;
    QHBoxLayout *horizontalLayout_3;
    QLabel *filterL;
    QComboBox *filterCB;
    QPushButton *filterPB;
    QSplitter *splitter;
    QGroupBox *ImageGB;
    QVBoxLayout *verticalLayout;
    TextureWidget *tex_w;
    QGroupBox *StackGB;
    QVBoxLayout *verticalLayout_2;
    QScrollArea *stackSCL;
    QWidget *scrollAreaWidgetContents;
    QVBoxLayout *stack_VL;
    QCheckBox *test;
    QMenuBar *menuBar;
    QMenu *menuOpen;
    QStatusBar *statusBar;

    void setupUi(QMainWindow *MainWindow)
    {
        if (MainWindow->objectName().isEmpty())
            MainWindow->setObjectName(QStringLiteral("MainWindow"));
        MainWindow->resize(851, 619);
        actionOpen = new QAction(MainWindow);
        actionOpen->setObjectName(QStringLiteral("actionOpen"));
        actionSave = new QAction(MainWindow);
        actionSave->setObjectName(QStringLiteral("actionSave"));
        centralWidget = new QWidget(MainWindow);
        centralWidget->setObjectName(QStringLiteral("centralWidget"));
        verticalLayout_3 = new QVBoxLayout(centralWidget);
        verticalLayout_3->setSpacing(6);
        verticalLayout_3->setContentsMargins(11, 11, 11, 11);
        verticalLayout_3->setObjectName(QStringLiteral("verticalLayout_3"));
        widget = new QWidget(centralWidget);
        widget->setObjectName(QStringLiteral("widget"));
        horizontalLayout_4 = new QHBoxLayout(widget);
        horizontalLayout_4->setSpacing(1);
        horizontalLayout_4->setContentsMargins(11, 11, 11, 11);
        horizontalLayout_4->setObjectName(QStringLiteral("horizontalLayout_4"));
        horizontalLayout_4->setContentsMargins(1, 1, 1, 1);
        compGB = new QGroupBox(widget);
        compGB->setObjectName(QStringLiteral("compGB"));
        horizontalLayout_2 = new QHBoxLayout(compGB);
        horizontalLayout_2->setSpacing(5);
        horizontalLayout_2->setContentsMargins(11, 11, 11, 11);
        horizontalLayout_2->setObjectName(QStringLiteral("horizontalLayout_2"));
        horizontalLayout_2->setContentsMargins(5, 5, 5, 5);
        compL = new QLabel(compGB);
        compL->setObjectName(QStringLiteral("compL"));

        horizontalLayout_2->addWidget(compL);

        compCB = new QComboBox(compGB);
        compCB->setObjectName(QStringLiteral("compCB"));

        horizontalLayout_2->addWidget(compCB);


        horizontalLayout_4->addWidget(compGB);

        filterGB = new QGroupBox(widget);
        filterGB->setObjectName(QStringLiteral("filterGB"));
        horizontalLayout_3 = new QHBoxLayout(filterGB);
        horizontalLayout_3->setSpacing(5);
        horizontalLayout_3->setContentsMargins(11, 11, 11, 11);
        horizontalLayout_3->setObjectName(QStringLiteral("horizontalLayout_3"));
        horizontalLayout_3->setContentsMargins(5, 5, 5, 5);
        filterL = new QLabel(filterGB);
        filterL->setObjectName(QStringLiteral("filterL"));

        horizontalLayout_3->addWidget(filterL);

        filterCB = new QComboBox(filterGB);
        filterCB->setObjectName(QStringLiteral("filterCB"));
        filterCB->setStyleSheet(QStringLiteral(""));

        horizontalLayout_3->addWidget(filterCB);

        filterPB = new QPushButton(filterGB);
        filterPB->setObjectName(QStringLiteral("filterPB"));
        QSizePolicy sizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(filterPB->sizePolicy().hasHeightForWidth());
        filterPB->setSizePolicy(sizePolicy);
        filterPB->setMinimumSize(QSize(0, 0));

        horizontalLayout_3->addWidget(filterPB);


        horizontalLayout_4->addWidget(filterGB);


        verticalLayout_3->addWidget(widget);

        splitter = new QSplitter(centralWidget);
        splitter->setObjectName(QStringLiteral("splitter"));
        QSizePolicy sizePolicy1(QSizePolicy::Expanding, QSizePolicy::Expanding);
        sizePolicy1.setHorizontalStretch(0);
        sizePolicy1.setVerticalStretch(0);
        sizePolicy1.setHeightForWidth(splitter->sizePolicy().hasHeightForWidth());
        splitter->setSizePolicy(sizePolicy1);
        splitter->setOrientation(Qt::Horizontal);
        ImageGB = new QGroupBox(splitter);
        ImageGB->setObjectName(QStringLiteral("ImageGB"));
        QSizePolicy sizePolicy2(QSizePolicy::Expanding, QSizePolicy::Preferred);
        sizePolicy2.setHorizontalStretch(0);
        sizePolicy2.setVerticalStretch(0);
        sizePolicy2.setHeightForWidth(ImageGB->sizePolicy().hasHeightForWidth());
        ImageGB->setSizePolicy(sizePolicy2);
        verticalLayout = new QVBoxLayout(ImageGB);
        verticalLayout->setSpacing(6);
        verticalLayout->setContentsMargins(11, 11, 11, 11);
        verticalLayout->setObjectName(QStringLiteral("verticalLayout"));
        verticalLayout->setContentsMargins(-1, 20, -1, -1);
        tex_w = new TextureWidget(ImageGB);
        tex_w->setObjectName(QStringLiteral("tex_w"));
        filterGB->raise();
        compGB->raise();

        verticalLayout->addWidget(tex_w);

        splitter->addWidget(ImageGB);
        StackGB = new QGroupBox(splitter);
        StackGB->setObjectName(QStringLiteral("StackGB"));
        QSizePolicy sizePolicy3(QSizePolicy::Preferred, QSizePolicy::Minimum);
        sizePolicy3.setHorizontalStretch(0);
        sizePolicy3.setVerticalStretch(0);
        sizePolicy3.setHeightForWidth(StackGB->sizePolicy().hasHeightForWidth());
        StackGB->setSizePolicy(sizePolicy3);
        StackGB->setMinimumSize(QSize(300, 0));
        verticalLayout_2 = new QVBoxLayout(StackGB);
        verticalLayout_2->setSpacing(6);
        verticalLayout_2->setContentsMargins(11, 11, 11, 11);
        verticalLayout_2->setObjectName(QStringLiteral("verticalLayout_2"));
        verticalLayout_2->setContentsMargins(-1, 20, -1, -1);
        stackSCL = new QScrollArea(StackGB);
        stackSCL->setObjectName(QStringLiteral("stackSCL"));
        QSizePolicy sizePolicy4(QSizePolicy::Minimum, QSizePolicy::Preferred);
        sizePolicy4.setHorizontalStretch(0);
        sizePolicy4.setVerticalStretch(0);
        sizePolicy4.setHeightForWidth(stackSCL->sizePolicy().hasHeightForWidth());
        stackSCL->setSizePolicy(sizePolicy4);
        stackSCL->setWidgetResizable(true);
        scrollAreaWidgetContents = new QWidget();
        scrollAreaWidgetContents->setObjectName(QStringLiteral("scrollAreaWidgetContents"));
        scrollAreaWidgetContents->setGeometry(QRect(0, 0, 645, 424));
        stack_VL = new QVBoxLayout(scrollAreaWidgetContents);
        stack_VL->setSpacing(6);
        stack_VL->setContentsMargins(11, 11, 11, 11);
        stack_VL->setObjectName(QStringLiteral("stack_VL"));
        stack_VL->setContentsMargins(4, 4, 4, -1);
        test = new QCheckBox(scrollAreaWidgetContents);
        test->setObjectName(QStringLiteral("test"));

        stack_VL->addWidget(test);

        stackSCL->setWidget(scrollAreaWidgetContents);

        verticalLayout_2->addWidget(stackSCL);

        splitter->addWidget(StackGB);

        verticalLayout_3->addWidget(splitter);

        MainWindow->setCentralWidget(centralWidget);
        menuBar = new QMenuBar(MainWindow);
        menuBar->setObjectName(QStringLiteral("menuBar"));
        menuBar->setGeometry(QRect(0, 0, 851, 31));
        menuOpen = new QMenu(menuBar);
        menuOpen->setObjectName(QStringLiteral("menuOpen"));
        MainWindow->setMenuBar(menuBar);
        statusBar = new QStatusBar(MainWindow);
        statusBar->setObjectName(QStringLiteral("statusBar"));
        MainWindow->setStatusBar(statusBar);

        menuBar->addAction(menuOpen->menuAction());
        menuOpen->addAction(actionOpen);
        menuOpen->addAction(actionSave);

        retranslateUi(MainWindow);

        QMetaObject::connectSlotsByName(MainWindow);
    } // setupUi

    void retranslateUi(QMainWindow *MainWindow)
    {
        MainWindow->setWindowTitle(QApplication::translate("MainWindow", "MainWindow", 0));
        actionOpen->setText(QApplication::translate("MainWindow", "Open", 0));
        actionSave->setText(QApplication::translate("MainWindow", "Save", 0));
        compGB->setTitle(QString());
        compL->setText(QApplication::translate("MainWindow", "Computation:", 0));
        compCB->clear();
        compCB->insertItems(0, QStringList()
         << QApplication::translate("MainWindow", "Serial", 0)
         << QApplication::translate("MainWindow", "TBB", 0)
         << QApplication::translate("MainWindow", "CUDA", 0)
        );
        filterGB->setTitle(QString());
        filterL->setText(QApplication::translate("MainWindow", "Filters:", 0));
        filterPB->setText(QApplication::translate("MainWindow", "ADD", 0));
        ImageGB->setTitle(QApplication::translate("MainWindow", "Image", 0));
        StackGB->setTitle(QApplication::translate("MainWindow", "Stack", 0));
        test->setText(QApplication::translate("MainWindow", "CheckBox", 0));
        menuOpen->setTitle(QApplication::translate("MainWindow", "File", 0));
    } // retranslateUi

};

namespace Ui {
    class MainWindow: public Ui_MainWindow {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_BASE_WINDOW_H
