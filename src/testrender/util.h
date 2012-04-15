#pragma once

#include <OpenImageIO/imageio.h>

OSL_NAMESPACE_ENTER

struct Counter {
    Counter(OIIO::ErrorHandler& errhandler, int max, const char* task) :
        errhandler(errhandler), m(), counter(0), max(max),
        last_percent_printed(-1), task(task) {}

    int getnext(int& value) {
        OIIO::lock_guard guard(m);
        value = counter++;
        if (value >= max) return false;
        int progress = 100 * value / (max - 1);
        if (last_percent_printed != progress && (progress % 5) == 0)
            errhandler.info("%s %2d%%", task, last_percent_printed = progress);
        return true;
    }

private:
    OIIO::ErrorHandler& errhandler;
    OIIO::mutex m;
    int counter, max, last_percent_printed;
    const char* task;
};

OSL_NAMESPACE_EXIT
