#ifndef STARTSCREEN_H
#define STARTSCREEN_H

#include <QWidget>
#include <QTimer>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

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

public slots:

    void start_stream();

private slots:

    void on_logIn_clicked();
    void render_frame();

private:
    Ui::StartScreen *ui;
    QTimer* image_timer;
    cv::VideoCapture mCapture;
};

#endif // STARTSCREEN_H
