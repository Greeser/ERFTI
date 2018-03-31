#ifndef WORKSCREEN_H
#define WORKSCREEN_H

#include <QWidget>
#include <QTimer>
#include <iostream>
#include "mainwindow.h"
#include "emotion/emotiw.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
namespace Ui {
class WorkScreen;
}

class WorkScreen : public QWidget
{
    Q_OBJECT

public:
    explicit WorkScreen(QWidget *parent = 0);
    ~WorkScreen();
    void start_stream();
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
private slots:
    void render_frame();

    void on_pushButton_clicked();

    void render_result(EmAndConf& eac);

private:
    Ui::WorkScreen *ui;
    QTimer* image_timer;
    cv::VideoCapture mCapture;
    cv::Mat current_frame_;
    std::unique_ptr<EmotiW> classifier_;
    pNet net_;
};

#endif // WORKSCREEN_H
