/*
Copyright (c) 2017 Sony Pictures Imageworks Inc., et al.
All Rights Reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

* Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.
* Neither the name of Sony Pictures Imageworks nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <algorithm>

#include <QAction>
#include <QLabel>
#include <QLayout>
#include <QString>

#include <OSL/oslconfig.h>
#include <OpenImageIO/imagebuf.h>
#include <OpenImageIO/strutil.h>

OIIO_NAMESPACE_BEGIN
namespace Strutil {
template<typename T> inline std::string to_string (const T& value);
template<> inline std::string to_string (const QString& value) { return value.toUtf8().data(); }
}  // end namespace Strutil
OIIO_NAMESPACE_END



OSL_NAMESPACE_ENTER
namespace QtUtils {


template <typename OBJ, typename ACT>
QAction* add_action (OBJ *obj, const std::string &name,
                     const std::string &label,
                     const std::string &hotkey="",
                     ACT trigger_action=nullptr)
{
    QAction *act = new QAction (label.size() ? label.c_str() : name.c_str(), obj);
    if (hotkey.size())
        act->setShortcut (QString(hotkey.c_str()));
    if (trigger_action)
        connect (act, &QAction::triggered, obj, trigger_action);
    return act;
}



// Remove and delete any widgets within the layout.
inline void
clear_layout (QLayout *lay)
{
    QLayoutItem *child;
    while ((child = lay->takeAt(0)) != 0) {
        if (child->widget())
            delete (child->widget());
        delete child;
    }
}



// Create a label whose text is given by printf-style arguments (type safe).
// The label will be "autotext" which means it will auto-detect RTF and
// render a subset of HTML formatting. For example, "<i>blah</i>" will
// render the 'blah' in italics.
template<typename... Args>
inline QLabel*
make_qlabel (string_view fmt, const Args&... args)
{
    std::string text = OIIO::Strutil::format (fmt, args...);
    auto label = new QLabel (text.c_str());
    label->setTextFormat (Qt::AutoText);
    return label;
}



// Make a QImage that "wraps" the pixels of the ImageBuf. The ImageBuf MUST
//  * Be 3 or 4 channel, uint8 pixels
//  * Outlive the lifetime of the QImage, without having anything done
//    to it that will reallocate its memory.
inline QImage
ImageBuf_to_QImage (const OIIO::ImageBuf& ib)
{
    using namespace OIIO;
    if (ib.storage() == ImageBuf::UNINITIALIZED)
        return QImage();
    const ImageSpec &spec (ib.spec());
    if (spec.nchannels != 3 || spec.format != TypeDesc::UINT8)
        return QImage();
    if (ib.cachedpixels())
        const_cast<ImageBuf*>(&ib)->make_writeable (true);
    return QImage ((const uchar*)ib.localpixels(),
                   spec.width, spec.height, (int)spec.scanline_bytes(),
                   QImage::Format_RGB888);
}



/// QtUtils::DoubleSpinBox is an improved QDoubleSpinBox with the following
/// enhancements:
/// * Default range is -1e6 to 1e6 (allows negatives and bigger values than
///   regular QDoubleSpinBox, unless you restrict the range further).
/// * Default single-step size is not fixed, but gets bigger or smaller
///   based on the current value. That makes it easier to make fine
///   adjustments especially near zero.
/// * KeyboardTracking turned off by default, so if you edit the value as
///   text, the change doesn't take effect until you hit enter or focus
///   on a different widget.
class DoubleSpinBox : public QDoubleSpinBox {
public:
    typedef QDoubleSpinBox parent_t;

    DoubleSpinBox (double val, QWidget *parent=nullptr) : QDoubleSpinBox(parent) {
        setValue (val);
        setDecimals (3);
        setMaximumWidth (100);
        setRange (-1e6, 1e6);
        setAccelerated (true);
        setKeyboardTracking (false);
        variable_step_size ();  // defaults
        set_step_size (value());
    }

#if 1
    virtual void stepBy (int steps) { // override to adjust step size
        set_step_size (value());
        parent_t::stepBy (steps);
    }
#endif

    void fixed_step_size (double step = 1.0) {
        m_variable_step_digits = 1;
        m_fixed_step = step;
    }

    void variable_step_size (int digits = 1, double minstep = 1e-3) {
        m_variable_step_digits = digits;
        m_variable_min_step = minstep;
    }

    void valueChanged (double v) {
        set_step_size (v);
        parent_t::valueChanged (v);
    }

private:
    void set_step_size (double valsize) {
        double ss = m_fixed_step;
        if (m_variable_step_digits) {
            ss = m_variable_min_step;
            double av = fabs(valsize);
            for (int d = 6; d > -4; --d) {
                double p = pow (10.0, double(d));
                if (av > p*1.1) {
                    ss = p*pow(0.1,double(m_variable_step_digits));
                    setDecimals (abs(d)+3);
                    break;
                }
            }
            ss = std::max (ss, m_variable_min_step);
        }
        setSingleStep (ss);
    }

    double m_fixed_step = 1.0;
    double m_variable_min_step = 1e-3;
    int m_variable_step_digits = 1;
};



} // end namespace QtUtils
OSL_NAMESPACE_EXIT
