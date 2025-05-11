#include "mainwindow.h"
#include "./ui_mainwindow.h"
#include <QScrollBar>
#include <QColorDialog>
#include <filesystem>
#include <chrono>

#if defined __has_include
#  if __has_include (<nvtx3/nvToolsExt.h>)
#    include <nvtx3/nvToolsExt.h>
#  endif
#endif

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    curr_scene = new scene();

    //buttons
    connect(ui->accept_filename, SIGNAL(clicked()), this, SLOT(acceptFilenameClicked()));
    connect(ui->button_save, SIGNAL(clicked()), this, SLOT(buttonSaveClicked()));
    connect(ui->button_edit, SIGNAL(clicked()), this, SLOT(buttonEditClicked()));

    //interactive mode
    connect(ui->spinBox_crop_left, SIGNAL(valueChanged(int)), this, SLOT(editParamsChanged()));
    connect(ui->spinBox_crop_right, SIGNAL(valueChanged(int)), this, SLOT(editParamsChanged()));
    connect(ui->spinBox_crop_bottom, SIGNAL(valueChanged(int)), this, SLOT(editParamsChanged()));
    connect(ui->spinBox_crop_top, SIGNAL(valueChanged(int)), this, SLOT(editParamsChanged()));
    connect(ui->doubleSpinBox_rotate, SIGNAL(valueChanged(double)), this, SLOT(editParamsChanged()));
    connect(ui->checkBox_reflectHorizontal, SIGNAL(clicked(bool)), this, SLOT(editParamsChanged()));
    connect(ui->checkBox_reflectVertical, SIGNAL(clicked(bool)), this, SLOT(editParamsChanged()));
    connect(ui->doubleSpinBox_shearX, SIGNAL(valueChanged(double)), this, SLOT(editParamsChanged()));
    connect(ui->doubleSpinBox_shearY, SIGNAL(valueChanged(double)), this, SLOT(editParamsChanged()));
}

void MainWindow::editParamsChanged()
{
    if (ui->checkBox_interactiveRender->isChecked())
    {
        this->buttonEditClicked();
    }
}

void MainWindow::buttonEditClicked()
{
    log("starting edit...");
    int op_i = 0;

    int crop_l = ui->spinBox_crop_left->value();
    int crop_r = ui->spinBox_crop_right->value();
    int crop_t = ui->spinBox_crop_top->value();
    int crop_b = ui->spinBox_crop_bottom->value();

    if (crop_l != 0 || crop_r != 0 || crop_t != 0 || crop_b != 0)
    {
        op_i++;
        log(QString::number(op_i) + ". crop");
        curr_scene->crop(crop_l, crop_t, crop_r, crop_b);
    }

    float rot_angle = ui->doubleSpinBox_rotate->value();
    if (rot_angle != 0)
    {
        op_i++;
        log(QString::number(op_i) + ". rotate");
        curr_scene->rotate(rot_angle);
    }

    bool reflectHorizontal = ui->checkBox_reflectHorizontal->isChecked();
    bool reflectVertical = ui->checkBox_reflectVertical->isChecked();
    if (reflectHorizontal || reflectVertical)
    {
        op_i++;
        log(QString::number(op_i) + ". reflect");
        curr_scene->reflect(reflectHorizontal, reflectVertical);
    }

    float shx = ui->doubleSpinBox_shearX->value();
    float shy = ui->doubleSpinBox_shearY->value();
    if (shx != 0 || shy != 0)
    {
        op_i++;
        log(QString::number(op_i) + ". shear");
        curr_scene->shear(shx, shy);
    }

    if (op_i > 0)
    {
        log("finishing edit, encoding...");
        curr_scene->encode();
        updateImage();
    }
}

void MainWindow::updateImage(bool imgChanged)
{
    //update curr_image
    if (imgChanged)
    {
        unsigned img_size = curr_scene->get_img_binary_size();
        unsigned char* img_binary = new unsigned char[img_size];

        curr_scene->get_img_binary(img_binary);

        std::string img_format_str;

        switch (curr_scene->get_codec()->native_format())
        {
            case PNG:
                img_format_str = "PNG";
                break;
            case JPEG:
                img_format_str = "JPG";
                break;
            default:
                log("unsupported image format, can't display");
                return;
        }

        curr_image = new QImage();
        curr_image->loadFromData(img_binary, img_size, img_format_str.c_str());

        delete [] img_binary;
    }

    if (curr_image == nullptr)
        return;

    QGraphicsScene *scene = new QGraphicsScene;

    scene->addPixmap(QPixmap::fromImage(*curr_image).scaled(ui->graphicsView_image->width(), ui->graphicsView_image->height(), Qt::KeepAspectRatio));
    ui->graphicsView_image->setScene(scene);
}

void MainWindow::acceptFilenameClicked()
{
    QString filename = ui->textinp_filename->text();
    log("accepted filename: " + filename);
    image_basename = std::filesystem::path(filename.toStdString()).stem();

    log("");
    log("starting reading...");

    curr_scene->load_image_file(filename.toStdString());
    curr_scene->decode();

    log("finished reading.");
    log("");

    updateImage();
}

void MainWindow::buttonSaveClicked()
{
    log("");
    if (image_basename == "")
    {
        log("nothing to save");
        log("");
        return;
    }

    std::string ext = "";

    switch (curr_scene->get_codec()->native_format())
    {
        case JPEG:
            ext = ".jpeg";
            break;
        case PNG:
            ext = ".png";
            break;
        default:
            log("unsuported image format, saving without extension");
    }

    std::string filepath = "output/"+ image_basename;
    log(QString::fromStdString("saving to file: " + filepath));

    curr_scene->save_image_file(filepath);
    log("saved.");

    log("");
}

void MainWindow::log(const QString txt)
{
    ui->label_log->setText(ui->label_log->text() + txt + "\n");
    ui->label_log->repaint();
    ui->scrollArea_log->verticalScrollBar()->setValue(ui->label_log->height());
}

void MainWindow::resizeEvent(QResizeEvent *event)
{
    updateImage(false);
}

MainWindow::~MainWindow()
{
    delete ui;
    delete curr_scene;
}
