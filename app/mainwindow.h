#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QThread>
#include "tracking/cascade_cnn.h"
namespace Ui {
class MainWindow;
}

enum class Screens {Start, Work} ;
using pNet = std::shared_ptr<FaceInception::CascadeCNN>;

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
    Screens screen_;
    pNet net_;
};

#endif // MAINWINDOW_H
