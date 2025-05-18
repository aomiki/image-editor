#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include "scene.h"

QT_BEGIN_NAMESPACE
namespace Ui {
class MainWindow;
}
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    void updateImage(bool imgChanged = true);
    ~MainWindow();

public slots:
    void acceptFilenameClicked();
    void buttonSaveClicked();
    void editParamsChanged();
    void buttonEditClicked();
    void grayscaleCheckboxClicked(bool checked);
    void blurSigmaChanged(double sigma);

private:
    scene* curr_scene;

    QImage* curr_image = nullptr;
    std::string image_basename = "";

    void log(const QString txt);
    Ui::MainWindow *ui;

protected:
    void resizeEvent(QResizeEvent *event);
};
#endif // MAINWINDOW_H
