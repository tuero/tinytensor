// storage_base.h
// Storage base class, each backend implements

#ifndef TINYTENSOR_STORAGE_BASE_H_
#define TINYTENSOR_STORAGE_BASE_H_

namespace tinytensor {

// This is shared amongst views of the same Array
class StorageBase {
public:
    StorageBase() = default;
    StorageBase(const StorageBase &) = default;
    StorageBase(StorageBase &&) = default;
    StorageBase &operator=(const StorageBase &) = default;
    StorageBase &operator=(StorageBase &&) = default;
    virtual ~StorageBase() = default;
};

}    // namespace tinytensor

#endif    // TINYTENSOR_STORAGE_BASE_H_
