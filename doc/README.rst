Documentation Source Layout
===========================

- all doc-related rst files are in **source**

  - the layout of **source** follows these rules:

    - :mod:`utensor_cgen` ralated rst files will organized
      as source modules' layout

      - ex: given **utensor_cgen.utils**, you should put the rst
        file in **source/utensor_cgen/utils.rst**
  - any other auxiliary/master rst files will be directly under
    **source** directory

    - ex: **index.rst**, **api.rst**, ...etc
