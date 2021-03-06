#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QThread>
#include "tracking/cascade_cnn.h"
namespace Ui {
class MainWindow;
}

enum class Screens {Start, Work, Sign} ;
using pNet = std::shared_ptr<FaceInception::CascadeCNN>;

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

private slots:
    void log_in(QString name);
    void sign_up();
    void registered();

private:
    Ui::MainWindow *ui;
    Screens screen_;
    pNet net_;
};

#endif // MAINWINDOW_H
