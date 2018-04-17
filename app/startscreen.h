#ifndef STARTSCREEN_H
#define STARTSCREEN_H

#include <QWidget>
#include <QTimer>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "mainwindow.h"
#include "qperson.h"

namespace Ui {
class StartScreen;
}

class StartScreen : public QWidget
{
    Q_OBJECT

public:
    explicit StartScreen(QWidget *parent = 0);
    bool setNet(pNet net) {
        if (net)
        {
            net_ = net;
            FrameFeatures::init(net_);
            return true;
        }
        else
        {
            std::cerr<<"Didn't transferred to startscreen \n";
            return false;
        }
    }
    ~StartScreen();

signals:
    void logged(QString name);

public slots:

    void start_stream();

private slots:

    void on_logIn_clicked();
    void render_frame();

    void on_signUp_clicked();

private:
    Ui::StartScreen *ui;
    QTimer* image_timer;
    cv::VideoCapture mCapture;
    pNet net_;
    QPersonSet persons_;

};

#endif // STARTSCREEN_H
