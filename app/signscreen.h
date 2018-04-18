#ifndef SIGNSCREEN_H
#define SIGNSCREEN_H

#include <QWidget>


#include <QWidget>
#include <QTimer>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "mainwindow.h"
#include "qperson.h"

namespace Ui {
class SignScreen;
}

class SignScreen : public QWidget
{
    Q_OBJECT

public:
    explicit SignScreen(QWidget *parent = 0);
    bool setNet(pNet net) {
        if (net)
        {
            net_ = net;
            return true;
        }
        else
        {
            std::cerr<<"Didn't transferred to startscreen \n";
            return false;
        }
    }
    ~SignScreen();

signals:
    void registered();

public slots:
    void start_stream();

private slots:
    void render_frame();

    void on_pushButton_clicked();

private:
    Ui::SignScreen *ui;
    QTimer* image_timer;
    cv::VideoCapture mCapture;
    cv::Mat current_frame_;
    pNet net_;
};

#endif // SIGNSCREEN_H
