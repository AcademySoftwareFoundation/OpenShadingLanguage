#pragma once

#include <atomic>
#include <thread>
#include <vector>

// Silly, simple and good enough parallel_for for the LUT
// generation tools
template<typename F>
void
parallel_for(unsigned begin, unsigned end, F c)
{
    struct Job {
        F callback;
        std::atomic<unsigned> begin;
        unsigned end;

        bool next()
        {
            unsigned i = begin++;
            if (i < end)
                callback(i);
            return i < end;
        }
    } j = { c, begin, end };
    std::vector<std::thread> threads;
    const int nt = std::max(1u, std::min(end - begin,
                                         std::thread::hardware_concurrency()));

    for (int i = 0; i < nt; ++i) {
        threads.emplace_back([&]() {
            do {
            } while (j.next());
        });
    }
    for (int i = 0; i < nt; ++i)
        threads[i].join();
}