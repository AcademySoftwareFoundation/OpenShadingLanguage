// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/imageworks/OpenShadingLanguage


#include "osltoyapp.h"
#include "codeeditor.h"

#include <QAction>
#include <QApplication>
#include <QCheckBox>
#include <QDir>
#include <QDoubleSpinBox>
#include <QErrorMessage>
#include <QFileDialog>
#include <QFontDatabase>
#include <QLabel>
#include <QLayout>
#include <QLineEdit>
#include <QMenu>
#include <QMenuBar>
#include <QMouseEvent>
#include <QPixmap>
#include <QPlainTextEdit>
#include <QPushButton>
#include <QScrollArea>
#include <QSlider>
#include <QSpinBox>
#include <QSplitter>
#include <QStatusBar>
#include <QStyleOptionSlider>
#include <QTabWidget>
#include <QTextEdit>
#include <QToolTip>

// QT's extension foreach defines a foreach macro which interferes
// with an OSL internal foreach method.  So we will undefine it here
#undef foreach
// It is recommended any uses of QT's foreach be migrated
// to use C++11 range based loops.

#include <OpenImageIO/errorhandler.h>
#include <OpenImageIO/filesystem.h>
#include <OpenImageIO/imagebuf.h>
#include <OpenImageIO/imagebufalgo.h>
#include <OpenImageIO/strutil.h>
#include <OpenImageIO/sysutil.h>
#include <OpenImageIO/thread.h>

#include <OSL/oslcomp.h>
#include <OSL/oslexec.h>
#include <OSL/oslquery.h>
#include "osltoyrenderer.h"
#include "qtutils.h"


OSL_NAMESPACE_ENTER
using namespace QtUtils;

class ValueSlider final : public QSlider {
public:
    ValueSlider(Qt::Orientation orientation, QWidget* parent)
        : QSlider(orientation, parent)
    {
        setTickPosition(QSlider::TicksRight);
    }

    void sliderChange(QAbstractSlider::SliderChange change)
    {
        QSlider::sliderChange(change);

        if (change == QAbstractSlider::SliderValueChange) {
            QStyleOptionSlider opt;
            initStyleOption(&opt);
            QRect handle = style()->subControlRect(QStyle::CC_Slider, &opt,
                                                   QStyle::SC_SliderHandle,
                                                   this);
            QPoint pos   = handle.topRight();

            QToolTip::showText(mapToGlobal(pos), QString::number(value()),
                               this);
        }
    }
};

static void
setBackground(QWidget* widget, QColor bg, QColor fg)
{
    QPalette palette = widget->palette();
    palette.setColor(widget->backgroundRole(), bg);
    palette.setColor(widget->foregroundRole(), fg);
    widget->setPalette(palette);
    widget->setAutoFillBackground(true);
}


struct PixelInfo {
    // The image and where to magnify
    const OIIO::ImageBuf& full;
    int x, y;
    Color3 Cd;
    Vec3 P;
    Vec2 uv;

    PixelInfo(const OIIO::ImageBuf& full) : full(full) {}
};


class Magnifier final : public QWidget {
    QLabel* m_image;
    QLabel* m_info;
    OSLToyRenderView* m_renderview;
    const int m_res;

public:
    void setInfo(const PixelInfo& info, OSLToyRenderView* owner, int pixels = 8)
    {
        m_renderview = owner;
        if (!m_renderview)
            return;

        OSL_DASSERT(m_image->width() == m_image->height());
        using namespace OIIO;

        int res      = m_image->width();
        float sample = float(pixels) * 0.5;
        ROI roi(info.x - sample, info.x + sample, info.y - sample,
                info.y + sample);

        ImageBuf display(ImageSpec(pixels, pixels, 3, TypeDesc::UINT8));
        ImageBuf cut;
        ImageBufAlgo::cut(cut, info.full, roi);
        ImageBufAlgo::colorconvert(display, cut, "linear", "sRGB");

        // Use Qt to do the image scaling, so minimal interpolation
        QImage qimage = QtUtils::ImageBuf_to_QImage(display).scaled(
            res, res, Qt::IgnoreAspectRatio);
        if (!qimage.isNull())
            m_image->setPixmap(QPixmap::fromImage(qimage));

        int fwidth = 0;
        char fmt   = 'f';
        m_info->setText(QString(" Cout:\n"
                                "  %1\n  %2\n  %3\n"
                                " P:\n"
                                "  %4\n  %5\n  %6\n"
                                " uv:\n"
                                "  %7\n  %8")
                            .arg(info.Cd.x, fwidth, fmt)
                            .arg(info.Cd.y, fwidth, fmt)
                            .arg(info.Cd.z, fwidth, fmt)
                            .arg(info.P.x, fwidth, fmt)
                            .arg(info.P.y, fwidth, fmt)
                            .arg(info.P.z, fwidth, fmt)
                            .arg(info.uv.x, fwidth, fmt)
                            .arg(info.uv.y, fwidth, fmt));
    }

    Magnifier(QWidget* parent, int res = 170)
        : QWidget(parent)
        , m_image(nullptr)
        , m_info(nullptr)
        , m_renderview(nullptr)
        , m_res(res)
    {
        int textSize = 100;
        setMinimumSize(m_res + textSize, m_res);
        setMaximumSize(m_res + textSize, m_res);

        QLayout* layout = new QHBoxLayout(this);
        setLayout(layout);

        m_image = new QLabel(this);
        m_image->setMinimumSize(m_res, m_res);
        m_image->setMaximumSize(m_res, m_res);
        m_image->setScaledContents(false);
        layout->addWidget(m_image);

        m_info = new QLabel(this);
        m_info->setAlignment(Qt::AlignTop | Qt::AlignLeft);
        m_info->setMinimumSize(textSize, m_res);
        layout->addWidget(m_info);

        setBackground(m_info, Qt::black, Qt::white);

        // Highlight the edges against solid backgrounds
        m_image->setStyleSheet("border: 2px solid black");

        // When mouse moves fast, it might get over the inspector.
        setAttribute(Qt::WA_Hover, true);
        setMouseTracking(true);
    }

    void hideEvent(QHideEvent* e) override
    {
        m_renderview = nullptr;
        QWidget::hideEvent(e);
    }

    void enterEvent(QEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
};

class OSLToyRenderView final : public QLabel {
    Magnifier* m_magnifier;
    OIIO::ImageBuf m_framebuffer;

    float static mapPixel(float pos, float res) { return (pos / res); }

    // Should probably live in PixelInfo, but its here now
    bool getPixel(QPoint pos, PixelInfo& info)
    {
        if (!m_framebuffer.initialized())
            return false;
        const auto& spec = m_framebuffer.spec();
        info.x           = pos.x();
        info.y           = pos.y();
        if (info.x > spec.width || info.y > spec.height)
            return false;
        info.uv = Vec2(mapPixel(pos.x(), spec.width),
                       mapPixel(pos.y(), spec.height));

        m_framebuffer.getpixel(info.x, info.y, &info.Cd.x, 3);
        info.P = Vec3(info.uv.x * spec.width, info.uv.y * spec.height, 0);
        return true;
    }

    static bool shouldShowMagnifier(const Qt::KeyboardModifiers& mods)
    {
        return mods.testFlag(Qt::AltModifier);
    }
    static bool shouldShowMagnifier()
    {
        return shouldShowMagnifier(QGuiApplication::keyboardModifiers());
    }

    void showMagnifier()
    {
        if (!m_magnifier) {
            // Create the magnifier in the top-most parent
            // so no siblings will draw on top
            QWidget* parent     = parentWidget();
            QWidget* lastParent = nullptr;
            do {
                lastParent = parent;
                parent     = parent->parentWidget();
            } while (parent);
            OSL_DASSERT(lastParent != nullptr);
            m_magnifier = new Magnifier(lastParent);
        }

        setCursor(Qt::CrossCursor);
        magnifierEvent(true);
        m_magnifier->show();
    }

public:
    OSLToyRenderView(int xres, int yres, QWidget* parent)
        : QLabel(parent), m_magnifier(nullptr)
    {
        // setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Ignored);
        // setAlignment (Qt::AlignHCenter | Qt::AlignVCenter);
        setScaledContents(false);
        setMinimumSize(xres, yres);
        setMaximumSize(xres, yres);

        setAttribute(Qt::WA_Hover, true);
        setMouseTracking(true);
    }

    // void hoverEnter(QHoverEvent* e) override;
    // void hoverLeave(QHoverEvent* e) override;
    // void hoverMove(QHoverEvent* e) override;

    void magnifierEvent(bool force = false)
    {
        if (force || shouldShowMagnifier()) {
            QPoint pos     = mapFromGlobal(QCursor::pos());
            PixelInfo info = { m_framebuffer };
            if (m_magnifier) {
                if (getPixel(pos, info)) {
                    m_magnifier->setInfo(info, this);
                    m_magnifier->move(pos + geometry().topLeft()
                                      + QPoint(40, 40));
                    m_magnifier->show();
                } else if (!m_magnifier->underMouse())
                    m_magnifier->hide();
            }
        } else if (m_magnifier)
            m_magnifier->hide();
    }

    void mouseMoveEvent(QMouseEvent* /*event*/) override { magnifierEvent(); }

    void enterEvent(QEvent* /*event*/) override
    {
        if (shouldShowMagnifier())
            showMagnifier();
    }

    void leaveEvent(QEvent* /*event*/) override
    {
        if (m_magnifier && !m_magnifier->underMouse())
            m_magnifier->hide();
    }

    bool update(const OIIO::ImageBuf& image)
    {
        using namespace OIIO;

        // Copy from the renderer's framebuffer to ours
        ImageBufAlgo::resize(m_framebuffer, image);

        // Copy from the renderer's framebuffer (linear float) to display (sRGB uint8)
        OIIO::ImageBuf display(
            ImageSpec(width(), height(), 3, TypeDesc::UINT8));
        ImageBufAlgo::colorconvert(display, image, "linear", "sRGB");
        QImage qimage = QtUtils::ImageBuf_to_QImage(display);
        if (qimage.isNull())
            return false;

        setPixmap(QPixmap::fromImage(qimage));
        return true;
    }

#if 0
    void keyPressEvent(QKeyEvent* event) override {
        if (shouldShowMagnifier(event->modifiers()))
            showMagnifier();
        else if (m_magnifier && m_magnifier->isVisible())
            m_magnifier->hide();
        event->setAccepted(true);
    }

    void keyReleaseEvent(QKeyEvent* event) override {
        if (m_magnifier && !shouldShowMagnifier() &&
            m_magnifier && m_magnifier->isVisible()) {
            m_magnifier->hide();
        }
    }
#endif
};

void
Magnifier::enterEvent(QEvent* event)
{
    if (m_renderview)
        m_renderview->magnifierEvent();
    else
        event->setAccepted(false);
}

void
Magnifier::mouseMoveEvent(QMouseEvent* event)
{
    if (m_renderview)
        m_renderview->magnifierEvent();
    else
        event->setAccepted(false);
}

// Shadertoy inspiration:
// ----------------------
// Shadertoy basic docs: https://www.shadertoy.com/howto
// Shadertoy key bindings:  https://shadertoyunofficial.wordpress.com/

// Qt Docs help:
// -------------
// Qt tutorials:  http://doc.qt.io/qt-5/qtexamplesandtutorials.html
// Qt code example for an image viewer:
//     http://doc.qt.io/qt-5/qtwidgets-widgets-imageviewer-example.html
// Qt code example for a code text editor with line numbers:
//     http://doc.qt.io/qt-5/qtwidgets-widgets-codeeditor-example.html
//
// QMainWindow:  http://doc.qt.io/qt-5/qmainwindow.html
// QLabel: http://doc.qt.io/qt-5/qlabel.html
// QPixmap: http://doc.qt.io/qt-5/qpixmap.html
// QImage: http://doc.qt.io/qt-5/qimage.html
// QSplitter: http://doc.qt.io/qt-5/qsplitter.html
// QPlainTextEdit: http://doc.qt.io/qt-5/qplaintextedit.html
//
//

OSLToyMainWindow::OSLToyMainWindow(OSLToyRenderer* rend, int xr, int yr)
    : QMainWindow(nullptr), xres(xr), yres(yr), m_renderer(rend)
{
    // read_settings ();

    setWindowTitle(tr("OSL Toy"));

    // Set size of the window
    // setFixedSize(100, 50);

    createActions();
    createMenus();
    createStatusBar();

    renderView = new OSLToyRenderView(xres, yres, this);
    m_renderer->set_resolution(xres, yres);

    QWidget* viewport     = new QWidget(this);
    QGridLayout* vplayout = new QGridLayout(this);
    viewport->setLayout(vplayout);
    vplayout->addWidget(renderView, 0, 0, 4, 1);
    vplayout->addWidget(new ValueSlider(Qt::Vertical, this), 0, 1, 4, 1);

    clear_param_area();  // will initialize i
    paramLayout->addWidget(new QLabel("Parameter Controls"), 0, 0);
    paramScroll->setWidget(paramWidget);
    paramScroll->setMinimumSize(700, 0);

    auto display_area        = new QWidget;
    auto display_area_layout = new QVBoxLayout;
    display_area->setLayout(display_area_layout);
    display_area_layout->addWidget(viewport);
    display_area_layout->addWidget(paramScroll);

    QPixmap pixmap(xres, yres);
    OIIO::ImageBuf checks(
        OIIO::ImageSpec(xres, yres, 3, OIIO::TypeDesc::UINT8));
    const float white[] = { 1, 1, 1 };
    const float black[] = { 0, 0, 0 };
    OIIO::ImageBufAlgo::checker(checks, 16, 16, 1, white, black);
    renderView->update(checks);

    textTabs = new QTabWidget;
    action_newfile();  // Start with one tab

    auto control_area        = new QWidget;
    auto control_area_layout = new QHBoxLayout;
    control_area->setLayout(control_area_layout);
    recompileButton = new QPushButton("Recompile");
    // recompileButton->setGeometry (10, 10, 80, 30);
    connect(recompileButton, &QPushButton::clicked, this,
            &OSLToyMainWindow::recompile_shaders);
    control_area_layout->addWidget(recompileButton);

    pauseButton = new QPushButton("Pause");
    connect(pauseButton, &QPushButton::clicked, this,
            &OSLToyMainWindow::toggle_pause);
    control_area_layout->addWidget(pauseButton);

    restartButton = new QPushButton("Restart");
    connect(restartButton, &QPushButton::clicked, this,
            &OSLToyMainWindow::restart_time);
    control_area_layout->addWidget(restartButton);

    auto editorarea = new QWidget;
    QFontMetrics fontmetrics(CodeEditor::fixedFont());
#if QT_VERSION >= QT_VERSION_CHECK(5, 13, 0)
    int Mwidth = fontmetrics.horizontalAdvance(QLatin1Char('M'));
#else
    // QFontMetric.width() deprecated from 5.11, marked as such in 5.13
    int Mwidth = fontmetrics.width(QLatin1Char('M'));
#endif
    editorarea->setMinimumSize(85 * Mwidth, 40 * fontmetrics.lineSpacing());
    auto editorarea_layout = new QVBoxLayout;
    editorarea->setLayout(editorarea_layout);
    editorarea_layout->addWidget(textTabs);
    editorarea_layout->addWidget(control_area);

    centralSplitter = new QSplitter(Qt::Horizontal);
    centralSplitter->addWidget(display_area);
    centralSplitter->addWidget(editorarea);
    centralSplitter->setStretchFactor(1, 1);
    setCentralWidget(centralSplitter);

    maintimer = new QTimer(this);
    maintimer->setInterval(10);
    connect(maintimer, &QTimer::timeout, this,
            &OSLToyMainWindow::timed_rerender_trigger);
    maintimer->start();
}



OSLToyMainWindow::~OSLToyMainWindow()
{
    // Make sure the shadingsys is destroyed before the renderer
    std::cout << shadingsys()->getstats(5) << "\n";
}



void
OSLToyMainWindow::createActions()
{
    add_action("Exit", "E&xit", "Ctrl+Q", &OSLToyMainWindow::finish_and_close);
    add_action("New file", "", "Ctrl+N", &OSLToyMainWindow::action_newfile);
    add_action("Open...", "", "Ctrl+O", &OSLToyMainWindow::action_open);
    add_action("Save", "", "Ctrl+S", &OSLToyMainWindow::action_save);
    add_action("Save As...", "", "Shift-Ctrl+S",
               &OSLToyMainWindow::action_saveas);
    add_action("Close File", "", "Ctrl+W", &OSLToyMainWindow::action_close);
    add_action("Edit Preferences...", "", "",
               &OSLToyMainWindow::action_preferences);
    add_action("About...", "", "", &OSLToyMainWindow::action_about);

    add_action("Copy", "", "Ctrl+C", &OSLToyMainWindow::action_copy);
    add_action("Cut", "", "Ctrl+X", &OSLToyMainWindow::action_cut);
    add_action("Paste", "", "Ctrl+V", &OSLToyMainWindow::action_paste);

    add_action("Recompile shaders", "", "Ctrl+R",
               &OSLToyMainWindow::recompile_shaders);
    add_action("Enter Full Screen", "", "",
               &OSLToyMainWindow::action_fullscreen);
}



void
OSLToyMainWindow::createMenus()
{
    // openRecentMenu = new QMenu(tr("Open recent..."), this);
    // for (auto& i : openRecentAct)
    //     openRecentMenu->addAction (i);

    fileMenu = new QMenu(tr("&File"), this);
    fileMenu->addAction(actions["New file"]);
    fileMenu->addAction(actions["Open..."]);
    // fileMenu->addMenu (openRecentMenu);
    fileMenu->addAction(actions["Save"]);
    fileMenu->addAction(actions["Save As..."]);
    fileMenu->addAction(actions["Close File"]);
    fileMenu->addSeparator();
    fileMenu->addAction(actions["Exit"]);
    fileMenu->addSeparator();
    fileMenu->addAction(actions["Edit Preferences..."]);
    menuBar()->addMenu(fileMenu);

    editMenu = new QMenu(tr("&Edit"), this);
    editMenu->addAction(actions["Copy"]);
    editMenu->addAction(actions["Cut"]);
    editMenu->addAction(actions["Paste"]);
    menuBar()->addMenu(editMenu);

    viewMenu = new QMenu(tr("&View"), this);
    viewMenu->addAction(actions["Enter Full Screen"]);
    menuBar()->addMenu(viewMenu);

    toolsMenu = new QMenu(tr("&Tools"), this);
    toolsMenu->addAction(actions["Recompile shaders"]);
    menuBar()->addMenu(toolsMenu);

    helpMenu = new QMenu(tr("&Help"), this);
    helpMenu->addAction(actions["About..."]);
    menuBar()->addMenu(helpMenu);
    // Bring up user's guide

    menuBar()->show();
}



void
OSLToyMainWindow::createStatusBar()
{
    statusFPS       = new QLabel;
    QFont fixedFont = QFontDatabase::systemFont(QFontDatabase::FixedFont);
    fixedFont.setPointSize(13);
    statusFPS->setFont(fixedFont);

    statusBar()->addWidget(statusFPS);
    update_statusbar_fps(0.0f, 0.0f);
}



ShadingSystem*
OSLToyMainWindow::shadingsys() const
{
    return m_renderer->shadingsys();
}



int
OSLToyMainWindow::ntabs() const
{
    int n = textTabs->count();
    return n;
}



void
OSLToyMainWindow::update_statusbar_fps(float time, float fps)
{
    statusFPS->setText(
        OIIO::Strutil::sprintf("  %.2f    FPS: %5.1f", time, fps).c_str());
}



void
OSLToyMainWindow::clear_param_area()
{
    if (paramScroll) {
        auto oldParamWidget = paramScroll->takeWidget();
        delete oldParamWidget;
    } else {
        paramScroll = new QScrollArea;
    }
    paramWidget = new QWidget;
    paramLayout = new QGridLayout;
    paramWidget->setLayout(paramLayout);
    // paramScroll->setWidget (paramWidget);
}



CodeEditor*
OSLToyMainWindow::add_new_editor_window(const std::string& filename)
{
    // Make the code editor itself
    auto texteditor = new CodeEditor(nullptr, filename);
    editors.push_back(texteditor);

    // Make an error display widget
    auto errdisplay = new QTextEdit;
    errdisplay->setReadOnly(true);
    errdisplay->setFixedHeight(50);
    errdisplay->setTextColor(Qt::black);
    errdisplay->setPlainText("Not yet compiled");
    error_displays.push_back(errdisplay);

    // Make a widget and layout to hold the code editor and error display
    auto editor_and_error_display = new QWidget;
    auto ed_err_layout            = new QVBoxLayout;
    editor_and_error_display->setLayout(ed_err_layout);
    ed_err_layout->addWidget(texteditor);
    ed_err_layout->addWidget(errdisplay);

    // Add the combo editor and error display as the contents of the tab
    int n = ntabs();
    if (filename.size()) {
        textTabs->addTab(editor_and_error_display,
                         texteditor->brief_filename().c_str());
    } else {
        std::string title = OIIO::Strutil::sprintf("untitled %d", n + 1);
        textTabs->addTab(editor_and_error_display, title.c_str());
    }
    textTabs->setCurrentIndex(n);
    return texteditor;
}



void
OSLToyMainWindow::action_newfile()
{
    add_new_editor_window();
}



static const char* s_file_filters = "Shaders (*.osl *.oslgroup);;"
                                    "All Files (*)";



void
OSLToyMainWindow::action_open()
{
    QStringList files = QFileDialog::getOpenFileNames(
        nullptr, "Select one or more files to open", QDir::currentPath(),
        s_file_filters, nullptr, QFileDialog::Option::DontUseNativeDialog);

    for (auto& name : files) {
        std::string filename = OIIO::Strutil::to_string(name);
        if (filename.empty())
            continue;
        open_file(filename);
    }
}



bool
OSLToyMainWindow::open_file(const std::string& filename)
{
    std::string contents;
    if (!OIIO::Filesystem::read_text_file(filename, contents))
        return false;

    int tab                = textTabs->currentIndex();
    CodeEditor* texteditor = editors[tab];
    bool current_tab_empty = (texteditor->blockCount() == 1)
                             && texteditor->text_string().empty();
    if (!current_tab_empty)
        texteditor = add_new_editor_window(filename);

    texteditor->set_filename(filename);
    texteditor->setPlainText(contents.c_str());
    textTabs->setTabText(textTabs->currentIndex(),
                         texteditor->brief_filename().c_str());
    return true;
}



void
OSLToyMainWindow::action_saveas()
{
    int tab                = textTabs->currentIndex();
    CodeEditor* texteditor = editors[tab];

    QString name;
    name = QFileDialog::getSaveFileName(
        nullptr, "Save buffer", texteditor->full_filename().c_str(),
        s_file_filters, nullptr, QFileDialog::Option::DontUseNativeDialog);
    if (name.isEmpty())
        return;
    texteditor->set_filename(name.toUtf8().data());
    textTabs->setTabText(tab, texteditor->brief_filename().c_str());
    action_save();
}



void
OSLToyMainWindow::action_save()
{
    int tab                = textTabs->currentIndex();
    CodeEditor* texteditor = editors[tab];
    std::string filename   = texteditor->full_filename();

    if (filename == "untitled" || filename == "") {
        action_saveas();
        return;
    }
    std::string text = texteditor->text_string();

    OIIO::ofstream out;
    OIIO::Filesystem::open(out, filename);
    if (out)
        out << text;
    if (out.fail()) {
        std::string msg = OIIO::Strutil::sprintf("Could not write %s",
                                                 filename);
        QErrorMessage err(nullptr);
        err.showMessage(msg.c_str());
        err.exec();
    }
}



// Separate thread pool just for the async render kickoff triggers, but use
// the default pool for the workers.
static OIIO::thread_pool trigger_pool;


void
OSLToyMainWindow::timed_rerender_trigger(void)
{
    if (paused)
        return;
    float now = timer();
    if (now - last_frame_update_time > 0.05f) {
        last_frame_update_time = now;
        update_statusbar_fps(now, fps);
    }
    if (!m_rerender_needed && !m_shader_uses_time)
        return;
    {
        OIIO::spin_lock lock(m_job_mutex);
        if (m_working)
            return;
        m_working = 1;
        renderer()->set_time(now);
    }
    trigger_pool.push([=](int) { this->osl_do_rerender(now); });
}



void
OSLToyMainWindow::osl_do_rerender(float /*frametime*/)
{
    using namespace OIIO;
    m_rerender_needed = 0;

    if (renderer()->shadergroup()) {
        float start = timer();
        renderer()->set_time(start);
        renderer()->render_image();
        OIIO_UNUSED_OK float rendertime = timer() - start;

        renderView->update(renderer()->framebuffer());

        float now = timer();
        // std::cout <<"render only " << (1.0f/rendertime) << "  with coco " << 1.0f/(now-start)
        //     << "   from last frame " << 1.0f / (now - last_finished_frametime) << "\n";
        spin_lock lock(m_job_mutex);
        if (now - last_fps_update_time > 0.5f) {
#if 1
            fps = 1.0f / (now - start);  // includes colorconvert
#else
            fps = 1.0f / rendertime;  // only shading time
#endif
            last_fps_update_time = now;
        }
        last_finished_frametime = now;
    }
    m_working = 0;
}



class MyOSLCErrorHandler final : public OIIO::ErrorHandler {
public:
    MyOSLCErrorHandler(OSLToyMainWindow* osltoy) : osltoy(osltoy) {}
    virtual void operator()(int /*errcode*/, const std::string& msg)
    {
        errors.emplace_back(msg);
    }
    void clear() { errors.clear(); }

    std::vector<std::string> errors;

private:
    OSLToyMainWindow* osltoy;
};



void
OSLToyMainWindow::recompile_shaders()
{
    m_groupspec.clear();
    m_firstshadername.clear();
    m_groupname.clear();

    QtUtils::clear_layout(paramLayout);
    bool ok = true;
    for (int tab = 0; tab < ntabs(); ++tab) {
        auto editor            = editors[tab];
        std::string briefname  = editor->brief_filename();
        std::string shadername = OIIO::Filesystem::filename(briefname);
        std::string source     = editor->text_string();
        if (OIIO::Strutil::ends_with(briefname, ".oslgroup")) {
            continue;
            // FIXME!  No current support for shader group specs

            m_groupname = shadername;
            m_groupspec = source;
            // This is the group!
        } else if (OIIO::Strutil::ends_with(briefname, ".osl")) {
            // This is a shader
            MyOSLCErrorHandler errhandler(this);
            OSLCompiler oslcomp(&errhandler);
            std::string osooutput;
            std::vector<std::string> options;
            ok = oslcomp.compile_buffer(source, osooutput, options, "",
                                        briefname);
            set_error_message(tab,
                              OIIO::Strutil::join(errhandler.errors, "\n"));
            if (ok) {
                // std::cout << osooutput << "\n";
                ok = shadingsys()->LoadMemoryCompiledShader(briefname,
                                                            osooutput);
                if (!ok) {
                    // FIXME -- handle .oso error. What can happen?
                }
            } else {
                // Force tab display to the error
                textTabs->setCurrentIndex(tab);
                break;
            }
            if (m_firstshadername.empty())
                m_firstshadername = shadername;

            // FIXME!  Only one shader currently!
            break;
        }
    }

    if (ok) {
        // If everything went ok so far, make a shader group
        build_shader_group();
        inventory_params();
        rebuild_param_area();
        if (paused && fps == 0 /* never started */)
            toggle_pause();
        rerender_needed();
    }
}



void
OSLToyMainWindow::toggle_pause()
{
    OIIO::spin_lock lock(m_job_mutex);
    if (paused) {
        pauseButton->setText("Pause");
        timer.start();
        paused = false;
    } else {
        pauseButton->setText("Continue");
        timer.stop();
        paused = true;
    }
    repaint();  //qApp->processEvents();
}



void
OSLToyMainWindow::restart_time()
{
    OIIO::spin_lock lock(m_job_mutex);
    timer.reset();
    timer.start();
    last_fps_update_time    = -1.0f;
    last_frame_update_time  = -1.0f;
    last_finished_frametime = -1.0f;
    update_statusbar_fps(0.0f, 0.0f);
}



void
OSLToyMainWindow::build_shader_group()
{
    // std::cout << "Rebuilding group\n";
    ShadingSystem* ss = renderer()->shadingsys();
    ShaderGroupRef group;
    if (m_groupspec.size()) {
        group = ss->ShaderGroupBegin(m_groupname, "surface", m_groupspec);
        ss->ShaderGroupEnd();
    } else if (m_firstshadername.size()) {
        group = ss->ShaderGroupBegin();
        for (auto&& instparam : m_shaderparam_instvalues) {
            ss->Parameter(instparam.name(), instparam.type(), instparam.data(),
                          !m_diddlers[instparam.name().string()]);
        }
        ss->Shader("surface", m_firstshadername, "layer1");
        ss->ShaderGroupEnd();
    }
    renderer()->set_shadergroup(group);

    m_shader_uses_time            = false;
    int num_globals_needed        = 0;
    const ustring* globals_needed = nullptr;
    ss->getattribute(group.get(), "num_globals_needed", num_globals_needed);
    ss->getattribute(group.get(), "globals_needed", TypeDesc::PTR,
                     &globals_needed);
    for (int i = 0; i < num_globals_needed; ++i)
        if (globals_needed[i] == "time")
            m_shader_uses_time = true;

    rerender_needed();
}



void
OSLToyMainWindow::inventory_params()
{
    ShadingSystem* ss    = renderer()->shadingsys();
    ShaderGroupRef group = renderer()->shadergroup();
    if (!group)
        return;

    int nlayers = 0;
    ss->getattribute(group.get(), "num_layers", nlayers);
    std::vector<ustring> layernames(nlayers);
    ss->getattribute(group.get(), "layer_names",
                     TypeDesc(TypeDesc::STRING, nlayers), &layernames[0]);
    m_shaderparams.clear();
    for (int i = 0; i < nlayers; ++i) {
        OSLQuery oslquery(group.get(), i);
        for (size_t p = 0; p < oslquery.nparams(); ++p) {
            auto param = oslquery.getparam(p);
            OSL_DASSERT(param);
            m_shaderparams.push_back(std::make_shared<ParamRec>(*param));
            m_shaderparams.back()->layername = layernames[i];
        }
    }
}



void
OSLToyMainWindow::make_param_adjustment_row(ParamRec* param,
                                            QGridLayout* layout, int row)
{
    auto diddleCheckbox = new QCheckBox("  ");
    if (m_diddlers[param->name.string()])
        diddleCheckbox->setCheckState(Qt::Checked);
    connect(diddleCheckbox, &QCheckBox::stateChanged, this,
            [=](int state) { set_param_diddle(param, state); });
    layout->addWidget(diddleCheckbox, row, 0);

    std::string typetext(param->type.c_str());
    if (param->isclosure)
        typetext = OIIO::Strutil::sprintf("closure %s", typetext);
    if (param->isstruct)
        typetext = OIIO::Strutil::sprintf("struct %s", param->structname);
    if (param->isoutput)
        typetext = OIIO::Strutil::sprintf("output %s", typetext);
    //    auto typeLabel = QtUtils::make_qlabelf ("<i>%s</i>", typetext);
    //    layout->addWidget (typeLabel, row, 1);
    auto nameLabel = new QLabel(
        OIIO::Strutil::sprintf("<i>%s</i>&nbsp;  <b>%s</b>", typetext,
                               param->name)
            .c_str());
    nameLabel->setTextFormat(Qt::RichText);
    layout->addWidget(nameLabel, row, 1);

    param->widgets.clear();
    if (param->type == TypeDesc::INT) {
        auto adjustWidget = new QSpinBox();
        adjustWidget->setValue(param->idefault[0]);
        adjustWidget->setRange(-10000, 10000);
        adjustWidget->setMaximumWidth(100);
        adjustWidget->setKeyboardTracking(false);
        layout->addWidget(adjustWidget, row, 2);
        param->widgets.push_back(adjustWidget);
        connect<void (QSpinBox::*)(int)>(adjustWidget, &QSpinBox::valueChanged,
                                         this, [=](int) {
                                             set_param_instance_value(param);
                                         });
    } else if (param->type == TypeDesc::FLOAT) {
        auto adjustWidget = new QtUtils::DoubleSpinBox(param->fdefault[0]);
        layout->addWidget(adjustWidget, row, 2);
        param->widgets.push_back(adjustWidget);
        connect<void (QDoubleSpinBox::*)(double)>(
            adjustWidget, &QDoubleSpinBox::valueChanged, this,
            [=](double) { set_param_instance_value(param); });
    } else if (param->type.is_vec3()) {
        auto xyzBox    = new QWidget;
        auto xyzLayout = new QHBoxLayout;
        xyzBox->setLayout(xyzLayout);
        xyzLayout->setSpacing(1);
        for (int c = 0; c < 3; ++c) {
            auto label_and_adjust        = new QWidget;
            auto label_and_adjust_layout = new QHBoxLayout;
            label_and_adjust->setLayout(label_and_adjust_layout);
            std::string labeltext;
            if (param->type == TypeDesc::TypeColor)
                labeltext = string_view(&("RGB"[c]), 1);
            else
                labeltext = string_view(&("xyz"[c]), 1);
            auto channellabel = QtUtils::make_qlabelf("%s", labeltext);
            label_and_adjust_layout->addWidget(channellabel);
            auto adjustWidget = new QtUtils::DoubleSpinBox(param->fdefault[c]);
            if (param->type == TypeDesc::TypeColor) {
                adjustWidget->setRange(0.0, 1.0);
            }
            label_and_adjust_layout->addWidget(adjustWidget);
            xyzLayout->addWidget(label_and_adjust);
            param->widgets.push_back(adjustWidget);
            connect<void (QDoubleSpinBox::*)(double)>(
                adjustWidget, &QDoubleSpinBox::valueChanged, this,
                [=](double) { set_param_instance_value(param); });
        }
        layout->addWidget(xyzBox, row, 2);
    } else if (param->type == TypeDesc::STRING) {
        auto adjustWidget = new QLineEdit();
        adjustWidget->setText(param->sdefault[0].c_str());
        layout->addWidget(adjustWidget, row, 2);
        param->widgets.push_back(adjustWidget);
        connect(adjustWidget, &QLineEdit::returnPressed, this,
                [=]() { set_param_instance_value(param); });
    }

    auto resetButton = new QPushButton("Reset");
    connect(resetButton, &QPushButton::clicked, this,
            [=]() { reset_param_to_default(param); });
    layout->addWidget(resetButton, row, 3);

    set_ui_to_paramval(param);
}



void
OSLToyMainWindow::set_ui_to_paramval(ParamRec* param)
{
    // Erase the instance value override
    auto found                 = m_shaderparam_instvalues.find(param->name);
    const OIIO::ParamValue* pv = (found != m_shaderparam_instvalues.end())
                                     ? &(*found)
                                     : nullptr;

    // Reset the value of the visible widget to the default
    if (param->type == TypeDesc::INT) {
        const int* val = pv ? (const int*)pv->data() : &param->idefault[0];
        reinterpret_cast<QSpinBox*>(param->widgets[0])->setValue(*val);
    } else if (param->type == TypeDesc::FLOAT) {
        const float* val = pv ? (const float*)pv->data() : &param->fdefault[0];
        reinterpret_cast<QDoubleSpinBox*>(param->widgets[0])->setValue(*val);
    } else if (param->type.is_vec3()) {
        const float* val = pv ? (const float*)pv->data() : &param->fdefault[0];
        for (int c = 0; c < 3; ++c) {
            reinterpret_cast<QDoubleSpinBox*>(param->widgets[c])
                ->setValue(val[c]);
        }
    } else if (param->type == TypeDesc::STRING) {
        const ustring* val = pv ? (const ustring*)pv->data()
                                : &param->sdefault[0];
        reinterpret_cast<QLineEdit*>(param->widgets[0])->setText(val->c_str());
    }
}



void
OSLToyMainWindow::reset_param_to_default(ParamRec* param)
{
    // Erase the instance value override
    auto instval = m_shaderparam_instvalues.find(param->name);
    if (instval != m_shaderparam_instvalues.end())
        m_shaderparam_instvalues.erase(instval);

    set_ui_to_paramval(param);
    build_shader_group();
}



void
OSLToyMainWindow::set_param_diddle(ParamRec* param, int diddle)
{
    m_diddlers[param->name.string()] = diddle;
    build_shader_group();
}



void
OSLToyMainWindow::set_param_instance_value(ParamRec* param)
{
    m_shaderparam_instvalues.remove(param->name);

    if (param->type == TypeDesc::INT) {
        int v = reinterpret_cast<QSpinBox*>(param->widgets[0])->value();
        m_shaderparam_instvalues.push_back(
            OIIO::ParamValue(param->name, param->type, 1, &v));
    } else if (param->type == TypeDesc::FLOAT) {
        float v = reinterpret_cast<QDoubleSpinBox*>(param->widgets[0])->value();
        m_shaderparam_instvalues.push_back(
            OIIO::ParamValue(param->name, param->type, 1, &v));
    } else if (param->type.is_vec3()) {
        float v[3];
        for (int c = 0; c < 3; ++c)
            v[c] = reinterpret_cast<QDoubleSpinBox*>(param->widgets[c])->value();
        m_shaderparam_instvalues.push_back(
            OIIO::ParamValue(param->name, param->type, 1, &v));
    } else if (param->type == TypeDesc::STRING) {
        std::string v = OIIO::Strutil::to_string(
            reinterpret_cast<QLineEdit*>(param->widgets[0])->text());
        m_shaderparam_instvalues.push_back(
            OIIO::ParamValue(param->name, param->type, 1, &v));
    }

    if (m_diddlers[param->name.string()]) {
        auto val = m_shaderparam_instvalues.find(param->name);
        if (val != m_shaderparam_instvalues.end())
            shadingsys()->ReParameter(*renderer()->shadergroup(),
                                      param->layername, param->name,
                                      param->type, val->data());
        rerender_needed();
    } else {
        build_shader_group();
    }
}



void
OSLToyMainWindow::rebuild_param_area()
{
    ShadingSystem* ss    = renderer()->shadingsys();
    ShaderGroupRef group = renderer()->shadergroup();
    if (!group)
        return;

    clear_param_area();
    int paramrow = 0;
    int nlayers  = 0;
    ss->getattribute(group.get(), "num_layers", nlayers);
    std::vector<ustring> layernames(nlayers);
    ss->getattribute(group.get(), "layer_names",
                     TypeDesc(TypeDesc::STRING, nlayers), &layernames[0]);
    for (int i = 0; i < nlayers; ++i) {
        OSLQuery oslquery(group.get(), i);
        std::string desc = OIIO::Strutil::sprintf("layer %d %s  (%s)", i,
                                                  layernames[i],
                                                  oslquery.shadername());
        paramLayout->addWidget(new QLabel(desc.c_str()), paramrow++, 0, 1, 2);
        for (auto&& p : m_shaderparams) {
            make_param_adjustment_row(p.get(), paramLayout, paramrow++);
        }
    }

    paramScroll->setWidget(paramWidget);
}



void
OSLToyMainWindow::set_error_message(int tab, const std::string& msg)
{
    OSL_DASSERT(tab >= 0 && tab < ntabs());
    if (msg.size()) {
        error_displays[tab]->setTextColor(Qt::red);
        error_displays[tab]->setPlainText(msg.c_str());
    } else {
        error_displays[tab]->setTextColor(Qt::darkGreen);
        error_displays[tab]->setPlainText("Ok");
    }
}



void
OSLToyMainWindow::finish_and_close()
{
    maintimer->setSingleShot(true);
    maintimer->setInterval(10000000);
    // wait for any shading jobs to finish
    for (; true; OIIO::Sysutil::usleep(10000)) {
        OIIO::spin_lock lock(m_job_mutex);
        if (m_working) {
            // If shading is still happening, release the lock and sleep
            // for 1/100 s.
            continue;
        }
        close();  // wrap it up for real
        break;
    }
}



void
OSLToyMainWindow::mousePressEvent(QMouseEvent* event)
{
#if 0
    // bool Alt = (event->modifiers() & Qt::AltModifier);
    // m_drag_button = event->button();
    switch (event->button()) {
    case Qt::LeftButton :
        std::cout << "Click " << event->x() << ' ' << event->y() << "\n";
        renderer()->set_mouse (event->x(), event->y());
        return;
    default:
        break;
    }
#endif
    QMainWindow::mousePressEvent(event);
}



OSL_NAMESPACE_EXIT
