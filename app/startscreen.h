#ifndef STARTSCREEN_H
#define STARTSCREEN_H

#include <QWidget>
#include "mainwindow.h"

namespace Ui {
class StartScreen;
}

class StartScreen : public QWidget
{
    Q_OBJECT

public:
    explicit StartScreen(QWidget *parent = 0);
    ~StartScreen();

signals:
    void logged();

private slots:

    void on_logIn_clicked();

private:
    Ui::StartScreen *ui;
};

#endif // STARTSCREEN_H
