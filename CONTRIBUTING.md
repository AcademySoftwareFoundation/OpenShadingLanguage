Contributing to OSL
===========================

Code contributions to OSL are always welcome. That's a big part of
why it's an open source project. Please review this document to get a
briefing on our process.


Mail List
---------

Contributors should be reading the osl-dev mail list:

* [osl-dev](http://groups.google.com/group/osl-dev)

You can sign up for the mail list on your own using the link above.


Bug Reports and Issue Tracking
------------------------------

We use GitHub's issue tracking system for bugs and enhancements:
https://github.com/imageworks/OpenShadingLanguage/issues

**If you are merely asking a question ("how do I...")**, please do not file an
issue, but instead ask the question on the [OSL developer mail
list](http://groups.google.com/group/osl-dev).

If you are submitting a bug report, please be sure to note which version of
OSL you are using, on what platform (OS/version, which compiler you used,
and any special build flags or other unusual environmental issues). Please
give an account of

* what you tried
* what happened
* what you expected to happen instead

with enough detail that others can reproduce the problem.


Contributor License Agreement (CLA) and Intellectual Property
-------------------------------------------------------------

To protect the project -- and the contributors! -- we do require a
Contributor License Agreement (CLA) for anybody submitting changes.

* If you are an individual writing the code on your own time and you're SURE
you are the sole owner of any intellectual property you contribute, use the
[Individual CLA](http://opensource.imageworks.com/cla/pdf/Imageworks_Contributor_License_Agreement_Individual.pdf).

* If you are writing the code as part of your job, or if there is any
possibility that your employers might think they own any intellectual
property you create, then you should use the [Corporate
CLA](http://opensource.imageworks.com/cla/pdf/Imageworks_Contributor_License_Agreement_Corporate.pdf).

Download the appropriate CLA from the links above, print, sign, and rescan
it (or just add a digital signature directly), and email it to:
opensource@imageworks.com

Our CLA's are based on those used by Apache and many other open source
projects.


Pull Requests and Code Review
-----------------------------

The best way to submit changes is via GitHub Pull Request. GitHub has a
[Pull Request Howto](https://help.github.com/articles/using-pull-requests/).

All code must be formally reviewed before being merged into the official
repository. The protocol is like this:

1. Get a GitHub account, fork imageworks/OpenShadingLanguage to create your
own repository on GitHub, and then clone it to get a repository on your
local machine.

2. Edit, compile, and test your changes.

3. Push your changes to your fork (each unrelated pull request to a separate
"topic branch", please).

4. Make a "pull request" on GitHub for your patch.

5. If your patch will induce a major compatibility break, or has a design
component that deserves extended discussion or debate among the wider OSL
community, then it may be prudent to email osl-dev pointing everybody to
the pull request URL and discussing any issues you think are important.

6. The reviewer will look over the code and critique on the "comments" area,
or discuss in email. Reviewers may ask for changes, explain problems they
found, congratulate the author on a clever solution, etc. But until somebody
says "LGTM" (looks good to me), the code should not be committed. Sometimes
this takes a few rounds of give and take. Please don't take it hard if your
first try is not accepted. It happens to all of us.

7. After approval, one of the senior developers (with commit approval to the
official main repository) will merge your fixes into the master branch.


Coding Style
------------

There are two overarching rules:

1. When making changes, conform to the style and conventions of the surrounding code.

2. Strive for clarity, even if that means occasionally breaking the
guidelines. Use your head and ask for advice if your common sense seems to
disagree with the conventions.

Below we try to enumerate the guidelines embodied in the code.

#### File conventions

C++ implementation should be named `*.cpp`. Headers should be named `.h`.
All headers should contain

    #pragma once


All source files should begin with the copyright and license, which can be
found in the [LICENSE](LICENSE) file (or cut and pasted from any other other source
file). Two notes on this:

* For NEW source files, please change the copyright year to the present. DO
NOT edit existing files only to update the copyright year, it just creates
pointless deltas and offers no increased protection.

* Occasionally a file contains substantial code from another project
and will also list its copyright. Do NOT copy that notice to any new files,
it really only applies to the particular file in which it appears.

#### Formatting

NEVER alter somebody else's code to reformat just because you found
something that violates the rules. Let the group/author/leader know, and
resist the temptation to change it yourself.

Each line of text in your code should be at most 80 characters long.
Exceptions are allowed for those rare cases where letting a line be longer
(and wrapping on an 80-character window) is actually a better and clearer
alternative than trying to split it into two lines. Sometimes this happens,
but it's extremely rare.

Indent 4 spaces at a time, and use actual spaces, not tabs. For files that
must use tabs for some reason, tabs should always be on 8's. Most editors
have a setting that forces indentations to be spaces. With emacs, you can do
this:

    (setq c-default-style "bsd")
    (setq-default indent-tabs-mode nil)

Opening brace on the same line as the condition or loop.

One statement per line, except in rare cases where violating this rule makes
the code more clear.

Three (3) consecutive empty lines between function or class method
implementations, one blank line between method declarations within a class
definition. Put a single blank line within a function if it helps to
visually separate different sequential tasks. Don't put multiple blank lines
in a row within a function, or blank lines right after an opening brace or
right before a closing brace. The goal is to use just enough white space to
help developers visually parse the code (for example, spotting at a glance
where new functions begin), but not so much as to spread it out or be
confusing.

For if, for, while, etc., put a space before the paren, but NOT inside the parens. For example:

    if (foo)    // Yes
    
    if(foo)     // No
    if ( foo )  // No

Function calls should have a space between the function name and the opening
parenthesis, NO space inside the parenthesis, except for a single blank
space between each argument. For example:

    x = foo (a, b);     // Yes, this is always ok
    
    x = foo ( a, b );   // No
    x = foo (a,b);      // No
    x = foo(a, b);      // No
    x = foo(a);         // Occasionally, this just looks better, when the function name is short,
                        //    and there's just one very short argument.  What can I say, I do it too.

Function declarations: function names should begin at column 0 for a full
function definition. (It's ok to violate this rule for very short inline
functions within class definitions.)


Here is a short code fragment that shows some of these rules in action:

    static int
    function (int a, int b)
    {
        int x = a + b;
        if (a == 0 || b == 0) {
            x += 1;
            x *= 4;
        } else {
            x -= 3;
        }
        for (int i = 0;  i < 3;  ++i) {
            x += a * i;
            x *= foo (i);  // function call
        }
        return x;
    }

Don't ever reformat, re-indent, change whitespace, or make any other such
changes to working code. If you're adding just a few lines or fixing a bug
in existing code, stick to the style of the surrounding code. In very rare
circumstances, and with consensus of the group, reformatting is sometimes
helpful to improve code readability, but it should be done as a separate
formatting-only checkin.

#### Identifiers

In general, classes and templates should start with upper case and capitalize new words:

    class CustomerList;

In general, local variables should start with lower case.

If your class is extremely similar to, or modeled after, something in the
standard library, Boost, or something else we interoperate with, it's ok to
use their naming conventions. For example, very general utility classes and
templates (the kind of thing you would normally find in std or boost) should
be lower case with underscores separating words, as they would be if they
were standards.

    template <class T> shared_ptr;
    class scoped_mutex;

Macros should be ALL_CAPS, if used at all.

Names of data should generally be nouns. Functions/methods are trickier: a
the name of a function that returns a value but has no side effects should
be a noun, but a procedure that performs an action should be a verb.

#### Class structure

Try to avoid public data members, although there are some classes that serve
a role similar to a simple C struct -- a very straightforward collection of
data members. In these, it's fine to make the data members public and have
clients set and get them directly.

Private member data should be named m_foo (alternately, it's ok to use the
common practice of member data foo_, but don't mix the conventions within a
class).

Private member data that needs public accessors should use the convention:

    void foo (const T& newfoo) { m_foo = newfoo; }
    const T& foo () const { return m_foo; }

Avoid multiple inheritance.

Namespaces: yes, use them!

#### Third-party libraries

Prefer C++11 `std` rather than Boost or other third party libraries, where
both can do the same task.

If you see a third party libary already used as a dependency (such as OIIO,
Boost, Ilmbase, or LLVM), feel free to any of its public features in OSL
internals (provided those features are present in the minimum version of
that library that we support).

Please do not add any NEW dependencies without debate on osl-dev and
approval of the project leader.

Use these libraries for OSL internals, but please DO NOT let them infect any
of our public APIs unless it's been thoroughly discussed and approved by the
group. (Exceptions: it's ok to use OIIO and Imath classes in our public
APIs.)

#### Comments and Doxygen

Comment philosophy: try to be clear, try to help teach the reader what's
going on in your code.

Prefer C++ comments (starting line with `//`) rather than C comments (`/* ... */`).

For any function that may be used by other programmers (e.g., public or
protected members of classes), please use Doxygen-style comments. They looks
like this:

    /// Explanation of a class.  Note THREE slashes!
    /// Also, you need at least two lines like this.  If you don't have enough
    /// for two lines, make one line blank like this:
    ///
    class myclass {
        ....
        float foo;  ///< Doxygen comments on same line look like this
    }

If you know Doxygen well, please feel free to use the various other markups.
But don't go so crazy with Doxygen markups that the comment itself, in an
ordinary editor, is not as readable as possible. The Doxygen-generated pages
are nice, but the place that needs to be most readable is in the code.

#### Miscellaneous

Macros should be used only rarely -- prefer inline functions, templates,
enums, or "const" values wherever possible.

#### Bottom Line

When in doubt, look elsewhere in the code base for examples of similar
structures and try to format your code in the same manner.
