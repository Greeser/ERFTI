#ifndef WORKSCREEN_H
#define WORKSCREEN_H

#include <QWidget>
#include <QTimer>
#include <iostream>
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

private slots:
    void render_frame();


private:
    Ui::WorkScreen *ui;
    QTimer* image_timer;
    cv::VideoCapture mCapture;
    std::unique_ptr<EmotiW> classifier_;
};

#endif // WORKSCREEN_H
