# Author: Christoph Lassner.

_dtype_str_translation = { 'int': 'i',
                           'i': 'i',
                           'float': 'f',
                           'f': 'f',
                           'double': 'd',
                           'd': 'd',
                           'ui': 'uint',
                           'uint': 'uint',
                           'unsigned int': 'uint',
                           'ui8': 'uint8',
                           'uint8': 'uint8',
                           'uint8_t': 'uint8',
                           'unsigned char': 'uint8',
                           'uchar': 'uint8',
                           'uc': 'uint8',
                           'char': 'int8',
                           'c': 'int8',
                           'int16': 'int16',
                           'int16_t': 'int16',
                           'i16': 'int16',
                           'uint16': 'uint16',
                           'uint16_t': 'uint16',
                           'ui16': 'uint16',
                           'std::vector<float>': 'fv',
                           'std::vector<double>': 'dv',
                           'std::pair<float,std::shared_ptr<std::vector<int16_t>>>': 'hp',
                           'std::vector<std::pair<float,std::shared_ptr<std::vector<int16_t>>>>': 'vhp',
                           'std::pair<std::shared_ptr<std::vector<float>>,std::shared_ptr<std::vector<float>>>': 'rpf',
                           'std::vector<std::pair<std::pair<std::shared_ptr<std::vector<float>>,std::shared_ptr<std::vector<float>>>,float>>': 'vprpff',
                           'std::pair<std::shared_ptr<std::vector<double>>,std::shared_ptr<std::vector<double>>>': 'rpd',
                           'std::vector<std::pair<std::pair<std::shared_ptr<std::vector<double>>,std::shared_ptr<std::vector<double>>>,float>>': 'vprpfd'  }

# Translations from C++ to C types.
# #include <stdint.h> is required!
_dtype_c_translation = {'int': 'int',
                        'void': 'void',
                        'float': 'float',
                        'double': 'double',
                        'uint': 'unsigned int',
                        'fertilized::uint': 'unsigned int',
                        'unsigned int': 'unsigned int',
                        'uint8': 'uint8_t',
                        'uint8_t': 'uint8_t',
                        'int16_t': 'int16_t',
                        'size_t': 'size_t', # otherwise unsigned long long int
                        'bool': 'int',
                        'std::string': 'char*'}

# See http://www.mathworks.de/help/matlab/apiref/mxcreatenumericarray.html.
_matlab_cpp_translation = {"mxDOUBLE_CLASS":"double",
                           "mxSINGLE_CLASS":"float",
                           "mxUINT64_CLASS":"uint64_t",
                           "mxINT64_CLASS":"int64_t",
                           "mxUINT32_CLASS":"uint32_t",
                           "mxINT32_CLASS":"int32_t",
                           "mxUINT16_CLASS":"uint16_t",
                           "mxINT16_CLASS":"int16_t",
                           "mxUINT8_CLASS":"uint8_t",
                           "mxINT8_CLASS":"int8_t"};
_cpp_matlab_translation = {}
for _key, _val in list(_matlab_cpp_translation.items()):
  _cpp_matlab_translation[_val] = _key
  if not _val in ['float', 'double']:
    _cpp_matlab_translation['std::'+_val] = _key
_cpp_matlab_translation['long'] = 'mxINT64_CLASS'
_cpp_matlab_translation['ulong'] = 'mxUINT64_CLASS'
_cpp_matlab_translation['int'] = 'mxINT32_CLASS'
_cpp_matlab_translation['uint'] = 'mxUINT32_CLASS'
_cpp_matlab_translation['fertilized::uint'] = 'mxUINT32_CLASS'
_cpp_matlab_translation['short'] = 'mxINT16_CLASS'
_cpp_matlab_translation['ushort'] = 'mxUINT16_CLASS'
_cpp_matlab_translation['char'] = 'mxINT8_CLASS'
_cpp_matlab_translation['uchar'] = 'mxUINT8_CLASS'
_cpp_matlab_translation['size_t'] = 'mxUINT64_CLASS'
