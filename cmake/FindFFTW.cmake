#[=======================================================================[.rst:
FindFFTW
--------

Finds the FFTW3 library and defines imported targets for it.

FFTW ships one library per precision, and optionally one threading library
per precision on top of that, so this module is component-based.

Components
^^^^^^^^^^

``Double``, ``Float``, ``LongDouble``
  The serial library for that precision.
``DoubleThreads``, ``FloatThreads``, ``LongDoubleThreads``
  The POSIX-threads companion library for that precision.
``DoubleOpenMP``, ``FloatOpenMP``, ``LongDoubleOpenMP``
  The OpenMP companion library for that precision.

With no components requested, ``Double``, ``Float`` and ``LongDouble`` are
searched for and all three are required.

Imported targets
^^^^^^^^^^^^^^^^

One target per found component, named after it: ``FFTW::Double``,
``FFTW::DoubleThreads`` and so on. Each carries the include directory, so
linking one is all a consumer needs. A threading target links its serial
counterpart, so ``FFTW::DoubleThreads`` alone is enough for threaded
double-precision use.

Result variables
^^^^^^^^^^^^^^^^

``FFTW_FOUND``
  True if every requested component was found.
``FFTW_INCLUDES``, ``FFTW_INCLUDE_DIRS``
  The directory holding ``fftw3.h``.
``FFTW_LIBRARIES``
  The serial libraries that were found. Retained for compatibility with
  callers written against the previous version of this module; prefer the
  imported targets.
``FFTW_<COMPONENT>_LIBRARY``
  The full path to each found component's library.

Hints
^^^^^

``FFTW_ROOT``, or the ``FFTWDIR`` environment variable
  Search exclusively under this prefix.
``FFTW_USE_STATIC_LIBS``
  Search only for static libraries.
#]=======================================================================]

if(NOT FFTW_ROOT AND DEFINED ENV{FFTWDIR})
  set(FFTW_ROOT $ENV{FFTWDIR})
endif()

# pkg-config gives us search paths on a typical Unix installation. It is a
# hint, not a requirement: a manually built FFTW often ships no .pc file.
find_package(PkgConfig QUIET)
if(PKG_CONFIG_FOUND AND NOT FFTW_ROOT)
  pkg_check_modules(PKG_FFTW QUIET "fftw3")
endif()

set(_fftw_suffixes_saved ${CMAKE_FIND_LIBRARY_SUFFIXES})
if(FFTW_USE_STATIC_LIBS)
  set(CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_STATIC_LIBRARY_SUFFIX})
endif()

# component -> library name
set(_fftw_component_names
    Double            fftw3
    Float             fftw3f
    LongDouble        fftw3l
    DoubleThreads     fftw3_threads
    FloatThreads      fftw3f_threads
    LongDoubleThreads fftw3l_threads
    DoubleOpenMP      fftw3_omp
    FloatOpenMP       fftw3f_omp
    LongDoubleOpenMP  fftw3l_omp)

if(NOT FFTW_FIND_COMPONENTS)
  set(FFTW_FIND_COMPONENTS Double Float LongDouble)
  foreach(_component IN LISTS FFTW_FIND_COMPONENTS)
    set(FFTW_FIND_REQUIRED_${_component} TRUE)
  endforeach()
endif()

# The header is shared by every precision.
if(FFTW_ROOT)
  find_path(FFTW_INCLUDES
            NAMES "fftw3.h"
            PATHS ${FFTW_ROOT}
            PATH_SUFFIXES "include"
            NO_DEFAULT_PATH)
else()
  find_path(FFTW_INCLUDES
            NAMES "fftw3.h"
            HINTS ${PKG_FFTW_INCLUDE_DIRS}
            PATHS ${INCLUDE_INSTALL_DIR})
endif()

set(FFTW_LIBRARIES)
foreach(_component IN LISTS FFTW_FIND_COMPONENTS)
  list(FIND _fftw_component_names ${_component} _index)
  if(_index EQUAL -1)
    if(FFTW_FIND_REQUIRED_${_component})
      message(FATAL_ERROR "FindFFTW: unknown component '${_component}'")
    endif()
    continue()
  endif()
  math(EXPR _index "${_index} + 1")
  list(GET _fftw_component_names ${_index} _library_name)

  if(FFTW_ROOT)
    find_library(FFTW_${_component}_LIBRARY
                 NAMES ${_library_name}
                 PATHS ${FFTW_ROOT}
                 PATH_SUFFIXES "lib" "lib64"
                 NO_DEFAULT_PATH)
  else()
    find_library(FFTW_${_component}_LIBRARY
                 NAMES ${_library_name}
                 HINTS ${PKG_FFTW_LIBRARY_DIRS}
                 PATHS ${LIB_INSTALL_DIR})
  endif()
  mark_as_advanced(FFTW_${_component}_LIBRARY)

  if(FFTW_${_component}_LIBRARY AND FFTW_INCLUDES)
    set(FFTW_${_component}_FOUND TRUE)
    if(NOT _component MATCHES "Threads$|OpenMP$")
      list(APPEND FFTW_LIBRARIES ${FFTW_${_component}_LIBRARY})
    endif()
  else()
    set(FFTW_${_component}_FOUND FALSE)
  endif()
endforeach()

set(CMAKE_FIND_LIBRARY_SUFFIXES ${_fftw_suffixes_saved})
unset(_fftw_suffixes_saved)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  FFTW
  REQUIRED_VARS FFTW_INCLUDES
  HANDLE_COMPONENTS)

if(FFTW_FOUND)
  set(FFTW_INCLUDE_DIRS ${FFTW_INCLUDES})

  # Two passes, so that a threading target can always find the serial target
  # it depends on regardless of the order the caller listed its components in.
  foreach(_pass IN ITEMS serial companion)
    foreach(_component IN LISTS FFTW_FIND_COMPONENTS)
      if(NOT FFTW_${_component}_FOUND OR TARGET FFTW::${_component})
        continue()
      endif()
      if(_component MATCHES "(Threads|OpenMP)$")
        if(_pass STREQUAL "serial")
          continue()
        endif()
      elseif(_pass STREQUAL "companion")
        continue()
      endif()

      add_library(FFTW::${_component} UNKNOWN IMPORTED GLOBAL)
      set_target_properties(
        FFTW::${_component}
        PROPERTIES IMPORTED_LOCATION "${FFTW_${_component}_LIBRARY}"
                   INTERFACE_INCLUDE_DIRECTORIES "${FFTW_INCLUDES}")

      # A threading library is a companion to, not a replacement for, the
      # serial library of the same precision, and must precede it on the link
      # line. Expressing that here spares every consumer from knowing it.
      if(_component MATCHES "^(.+)(Threads|OpenMP)$")
        set(_serial ${CMAKE_MATCH_1})
        set(_kind ${CMAKE_MATCH_2})
        if(TARGET FFTW::${_serial})
          set_property(TARGET FFTW::${_component}
                       APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                       FFTW::${_serial})
        endif()
        if(_kind STREQUAL "Threads")
          find_package(Threads QUIET)
          if(TARGET Threads::Threads)
            set_property(TARGET FFTW::${_component}
                         APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                         Threads::Threads)
          endif()
        else()
          find_package(OpenMP QUIET COMPONENTS CXX)
          if(TARGET OpenMP::OpenMP_CXX)
            set_property(TARGET FFTW::${_component}
                         APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                         OpenMP::OpenMP_CXX)
          endif()
        endif()
      endif()
    endforeach()
  endforeach()
endif()

mark_as_advanced(FFTW_INCLUDES)
