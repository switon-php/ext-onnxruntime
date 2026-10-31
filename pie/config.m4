PHP_ARG_WITH([onnxruntime],
  [for ONNX Runtime support],
  [AS_HELP_STRING([--with-onnxruntime=DIR],
    [ONNX Runtime install prefix])])

if test "$PHP_ONNXRUNTIME" != "no"; then
  if test "$PHP_ONNXRUNTIME" = "yes"; then
    PHP_ONNXRUNTIME="$abs_srcdir/../deps/onnxruntime-osx-arm64-1.31.1"
  fi

  AC_MSG_CHECKING([for ONNX Runtime in $PHP_ONNXRUNTIME])
  if test ! -f "$PHP_ONNXRUNTIME/include/onnxruntime_c_api.h"; then
    AC_MSG_ERROR([onnxruntime_c_api.h not found under $PHP_ONNXRUNTIME/include])
  fi
  if test ! -f "$PHP_ONNXRUNTIME/lib/libonnxruntime.dylib" && test ! -f "$PHP_ONNXRUNTIME/lib/libonnxruntime.so"; then
    AC_MSG_ERROR([libonnxruntime not found under $PHP_ONNXRUNTIME/lib])
  fi
  AC_MSG_RESULT([ok])

  PHP_ADD_INCLUDE([$PHP_ONNXRUNTIME/include])
  PHP_ADD_LIBRARY_WITH_PATH([onnxruntime], [$PHP_ONNXRUNTIME/lib], [ONNXRUNTIME_SHARED_LIBADD])
  ONNXRUNTIME_SHARED_LIBADD="$ONNXRUNTIME_SHARED_LIBADD -Wl,-rpath,$PHP_ONNXRUNTIME/lib"
  case $(uname -s) in
    Darwin) ONNXRUNTIME_SHARED_LIBADD="$ONNXRUNTIME_SHARED_LIBADD -Wl,-rpath,@loader_path" ;;
    *) ONNXRUNTIME_SHARED_LIBADD="$ONNXRUNTIME_SHARED_LIBADD -Wl,-rpath,\$\$ORIGIN" ;;
  esac
  PHP_SUBST([ONNXRUNTIME_SHARED_LIBADD])

  PHP_NEW_EXTENSION([onnxruntime], [onnxruntime.c value.c session.c], [$ext_shared])
fi
