#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>

namespace Ui {
class MainWindow;
}

enum class Screens {Start, Work} ;

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

private slots:
    void log_in();

private:
    Ui::MainWindow *ui;
    Screens screen;
};

#endif // MAINWINDOW_H
