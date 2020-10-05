// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/imageworks/OpenShadingLanguage

#pragma once

#include <atomic>
#include <unordered_map>

#include <OpenImageIO/imagebuf.h>
#include <OpenImageIO/paramlist.h>
#include <OpenImageIO/thread.h>
#include <OpenImageIO/timer.h>

#include <OSL/oslconfig.h>
#include <OSL/oslquery.h>

#include <QAction>
#include <QMainWindow>
#include <QTimer>

class QGridLayout;
class QLabel;
class QMenu;
class QMouseEvent;
class QPlainTextEdit;
class QPushButton;
class QScrollArea;
class QSplitter;
class QTextEdit;
class CodeEditor;

OSL_NAMESPACE_ENTER

class ShadingSystem;
class RendererServices;
class OSLToyRenderer;



class ParamRec final : public OSLQuery::Parameter {
public:
    using Parameter = OSLQuery::Parameter;
    // Inherits everything from OSLQuery::Parameter, and...
    std::vector<QWidget*> widgets;
    ustring layername;

    ParamRec() {}
    ParamRec(const Parameter& p) : Parameter(p) {}
    ParamRec(const ParamRec& p) : Parameter(p), widgets(p.widgets) {}
    ParamRec(ParamRec&& p) : Parameter(p), widgets(p.widgets) {}
};

class OSLToyRenderView;

class OSLToyMainWindow final : public QMainWindow {
    Q_OBJECT


public:
    explicit OSLToyMainWindow(OSLToyRenderer* rend, int xr, int yr);
    ~OSLToyMainWindow();

    void update_statusbar_fps(float time, float fps);

    void recompile_shaders();
    void build_shader_group();
    void toggle_pause();
    void restart_time();

    void finish_and_close();

    // Number of current active tabs
    int ntabs() const;

    // Replace the error message of a tab (empty message clears it)
    void set_error_message(int tab, const std::string& msg = "");

    // Create a new tab and its contained editor and message display, with
    // optional filename. Return the pointer to the CodeEditor widget.
    CodeEditor* add_new_editor_window(const std::string& filename = "");

    OSLToyRenderer* renderer() const { return m_renderer.get(); }

    ShadingSystem* shadingsys() const;

    bool open_file(const std::string& filename);

    void rerender_needed() { m_rerender_needed = 1; }

private slots:

private:
    int xres = 512;
    int yres = 512;

    // Non-owning pointers to all the widgets we create. Qt is responsible
    // for deleting.
    QSplitter* centralSplitter;
    OSLToyRenderView* renderView = nullptr;
    QTabWidget* textTabs         = nullptr;
    QScrollArea* paramScroll     = nullptr;
    QWidget* paramWidget         = nullptr;
    QGridLayout* paramLayout     = nullptr;
    QLabel* statusFPS;
    QMenu *fileMenu, *editMenu, *viewMenu, *toolsMenu, *helpMenu;
    QPushButton *recompileButton, *pauseButton, *restartButton;
    std::vector<CodeEditor*> editors;
    std::vector<QTextEdit*> error_displays;
    QTimer* maintimer;

    // Add an action, with optional label (if different than the name),
    // hotkey shortcut and the method of lambda to call when the action is
    // triggered. The QAction* is returned, but you don't need to store it;
    // it will also be saved in the actions map, accessed by name. Note that
    // to do anything fancier, like have actions on hover, set tooltips, or
    // whatever, you'll need to do that using the QAction*.
    // http://doc.qt.io/qt-5/qaction.html
    template<typename ACT>
    QAction* add_action(const std::string& name, const std::string& label,
                        const std::string& hotkey = "",
                        ACT trigger_action        = nullptr)
    {
        QAction* act  = new QAction(label.size() ? label.c_str() : name.c_str(),
                                   this);
        actions[name] = act;
        if (hotkey.size())
            act->setShortcut(QString(hotkey.c_str()));
        if (trigger_action)
            connect(act, &QAction::triggered, this, trigger_action);
        return act;
    }

    // Store all the actions in a list, accessed by name.
    std::unordered_map<std::string, QAction*> actions;

    // Create all the standard actions
    void createActions();

    // Create all the menu bar menus
    void createMenus();

    // Set up the status bar
    void createStatusBar();

    // Actions. To make these do things, put them in the .cpp and give them
    // bodies. Delete the ones that don't correspond to concepts in your
    // app.
    void action_newfile();
    void action_open();
    void action_close() {}
    void action_save();
    void action_saveas();
    void action_preferences() {}
    void action_copy() {}
    void action_cut() {}
    void action_paste() {}
    void action_hammer() {}
    void action_drill() {}
    void action_fullscreen() {}
    void action_about() {}

    void set_ui_to_paramval(ParamRec* param);
    void reset_param_to_default(ParamRec* param);
    void set_param_instance_value(ParamRec* param);
    void set_param_diddle(ParamRec* param, int state);

    virtual void mousePressEvent(QMouseEvent* event);

    void timed_rerender_trigger();

    void osl_do_rerender(float frametime);

    // Clear the param area. After this call, add things to paramLayout.
    // When you are done, call: paramScroll->setWidget (paramWidget)
    void clear_param_area();

    void make_param_adjustment_row(ParamRec* param, QGridLayout* layout,
                                   int row);

    void rebuild_param_area();
    void inventory_params();
    OIIO::ImageBuf& framebuffer();

    std::unique_ptr<OSLToyRenderer> m_renderer;

    std::vector<std::shared_ptr<ParamRec>> m_shaderparams;
    OIIO::ParamValueList m_shaderparam_instvalues;
    std::unordered_map<std::string, bool> m_diddlers;
    std::string m_groupspec;
    std::string m_firstshadername;
    std::string m_groupname;
    bool m_shader_uses_time = false;

    // Access control mutex for handing things off between the GUI thread
    // and the shading thread.
    OIIO::spin_mutex m_job_mutex;
    //vvv-- access these only if m_job_mutex is held
    std::atomic<int> m_working { 0 };
    std::atomic<int> m_shaders_recompiled { 0 };
    std::atomic<int> m_rerender_needed { 0 };
    //vvv--- access by the GUI thread only if m_working == 0, and by the
    //       shading thread only if m_working == 1.
    OIIO::Timer timer { false /*don't start*/ };
    float fps                     = 0;
    float last_frame_update_time  = -1;
    float last_fps_update_time    = -1;
    float last_finished_frametime = 0;
    bool paused                   = true;
};



OSL_NAMESPACE_EXIT
