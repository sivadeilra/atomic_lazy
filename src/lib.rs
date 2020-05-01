#![cfg_attr(all(not(test), not(feature = "std")), no_std)]

//! Yet Another Lazy Container
//!
//! `AtomicLazy` allows lazily computing and storing values in shared memory. It allows multiple
//! threads to access (safely) a shared value and potentially to move a value into that shared
//! location. `AtomicLazy` uses an atomic state variable, internally, to keep track of whether it
//! contains a value, or no value, or is currently being updated.
//!
//! # Why another type?
//!
//! Lazy initialization is not a new idea, and neither is lazy initialization using atomics.
//! The Rust ecosystem already has several types and crates for managing lazily-computed
//! values; one of these might already be a better fit for your needs:
//!
//! * `std::sync::Once`
//! * [`lazy_static`](https://crates.io/crates/lazy_static)
//! * [`lazy`](https://crates.io/crates/lazy)
//!
//! These are all good crates. However, they optimize for certain usage patterns, and make it
//! difficult or impossible to use in certain situations.
//!
//! * `atomic_lazy` can be used in `#![no_std]` environments.
//! * `atomic_lazy` provides an `unsafe` API which allows initializing large objects in-place.
//!   This `unsafe` interface can also be used to initialize objects that have mutual references.
//! * `atomic_lazy` can be used in shared contexts other than `&'static`.
//! * `atomic_lazy` does not combine the code which computes delayed values with the lazy cell
//!   itself. This allows for different code paths that initialize a single cell, and also does
//!   not store any data in a cell (relating to computing the on-demand value).
//!
//! # Author
//! * Arlie Davis

#![deny(missing_docs)]

use core::cell::UnsafeCell;
use core::mem::MaybeUninit;
use core::ptr::NonNull;
use core::sync::atomic::{AtomicU8, Ordering::SeqCst};

/// The AtomicLazy does not contain a value, and no thread is attempting to update it.
const LAZY_STATE_EMPTY: u8 = 0;

/// A thread is attempting to store a value into the AtomicLazy. Nothing must access the contents
/// during this time.
const LAZY_STATE_UPDATING: u8 = 1;

/// The contents of the AtomicLazy are now well-defined, and contain a value.
const LAZY_STATE_READY: u8 = 2;

/// A container that may contain a single value. A value can be placed inside `AtomicLazy`, but
/// cannot be removed. Moving an object into `AtomicLazy` is not atomic, so a separate `state`
/// field keeps track of whether a value is absent, currently being moved, or is present.
pub struct AtomicLazy<T> {
    cell: UnsafeCell<MaybeUninit<T>>,
    state: AtomicU8,
}

unsafe impl<T> Sync for AtomicLazy<T> {}

impl<T> Default for AtomicLazy<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> AtomicLazy<T> {
    /// Creates a new, empty `AtomicLazy`.
    ///
    /// ```
    /// # use atomic_lazy::AtomicLazy;
    /// static LAZY: AtomicLazy<String> = AtomicLazy::new();
    /// ```
    pub const fn new() -> Self {
        Self {
            cell: UnsafeCell::new(MaybeUninit::uninit()),
            state: AtomicU8::new(LAZY_STATE_EMPTY),
        }
    }

    /// Creates a new `AtomicLazy` that already contains a value.
    ///
    /// ```
    /// # use atomic_lazy::AtomicLazy;
    /// static LAZY: AtomicLazy<u32> = AtomicLazy::from_value(42);
    pub const fn from_value(value: T) -> Self {
        Self {
            cell: UnsafeCell::new(MaybeUninit::new(value)),
            state: AtomicU8::new(LAZY_STATE_READY),
        }
    }

    unsafe fn value_ptr(&self) -> *const T {
        (*self.cell.get()).as_ptr()
    }

    unsafe fn value_mut_ptr(&self) -> *mut T {
        (*self.cell.get()).as_mut_ptr()
    }

    unsafe fn value_ref(&self) -> &T {
        &*self.value_ptr()
    }

    /// Retrieves a reference to the contained value, or `None` if it is empty.
    pub fn get(&self) -> Option<&T>
    where
        T: Send,
    {
        let state = self.state.load(SeqCst);
        if state == LAZY_STATE_READY {
            unsafe { Some(self.value_ref()) }
        } else {
            None
        }
    }

    /// Gets or creates a contained value and returns a reference to it, using a provided function
    /// to compute the value if needed.
    ///
    /// If two (or more) threads call this concurrently, then it is possible for each of them to
    /// compute their own version of the value. Only one of them will "win" and have the value
    /// stored into the `AtomicLazy`; the rest of the values will be dropped.
    ///
    /// The advantage of using this method over `get_or_create_with_spin` is that it never blocks
    /// a calling thread. For situations where computing a value is relatively inexpensive, or
    /// where you know that concurrency is not an issue, this method is a good choice.
    ///
    /// ```
    /// # use atomic_lazy::AtomicLazy;
    /// static LAZY_VALUE: AtomicLazy<Option<String>> = AtomicLazy::new();
    ///
    /// LAZY_VALUE.get_or_create_with_dup(|| std::env::var("SOME_VAR_NAME").ok());
    /// ```
    pub fn get_or_create_with_dup<F: FnOnce() -> T>(&self, f: F) -> &T
    where
        T: Send,
    {
        let state = self.state.load(SeqCst);
        if state == LAZY_STATE_READY {
            unsafe {
                return self.value_ref();
            }
        }

        let value = f();
        loop {
            match self
                .state
                .compare_exchange(LAZY_STATE_EMPTY, LAZY_STATE_UPDATING, SeqCst, SeqCst)
            {
                Ok(_) => unsafe {
                    core::ptr::write(self.value_mut_ptr(), value);
                    self.state.store(LAZY_STATE_READY, SeqCst);
                    return self.value_ref();
                },
                Err(LAZY_STATE_UPDATING) => {
                    // Another thread is updating the lock. Release the current CPU and try again.
                    yield_thread();
                }
                Err(LAZY_STATE_READY) => {
                    // Another thread already finished updating the value; the value that we
                    // computed will not be used.
                    unsafe { return self.value_ref() }
                }
                Err(unrecognized_state) => {
                    panic!("Illegal state value {} in AtomicLazy", unrecognized_state)
                }
            }
        }
    }

    /// Gets or creates a contained value and returns a reference to it, using a provided function
    /// to compute the value if needed.
    ///
    /// If two (or more) threads call this concurrently, then it is possible for each of them to
    /// compute their own version of the value. Only one of them will "win" and have the value
    /// stored into the `AtomicLazy`; the rest of the values will be dropped.
    ///
    /// The advantage of using this method over `get_or_create_with_dup` is that it guarantees
    /// that only a single value will be computed. For situations where computing a value is very
    /// expensive, or has side-effects (such as reserving a TCP port), this method should be
    /// preferred.
    ///
    /// ```
    /// # use atomic_lazy::AtomicLazy;
    /// static LAZY_VALUE: AtomicLazy<Option<String>> = AtomicLazy::new();
    ///
    /// LAZY_VALUE.get_or_create_with_spin(|| std::env::var("SOME_VAR_NAME").ok());
    /// ```
    pub fn get_or_create_with_spin<F: FnOnce() -> T>(&self, f: F) -> &T
    where
        T: Send,
    {
        let state = self.state.load(SeqCst);
        if state == LAZY_STATE_READY {
            unsafe {
                return self.value_ref();
            }
        }

        loop {
            match self
                .state
                .compare_exchange(LAZY_STATE_EMPTY, LAZY_STATE_UPDATING, SeqCst, SeqCst)
            {
                Ok(_) => unsafe {
                    // This thread now "owns" the container.
                    struct Updater<'a, T> {
                        lazy: &'a AtomicLazy<T>,
                        next_state: u8,
                    }

                    impl<'a, T> Drop for Updater<'a, T> {
                        fn drop(&mut self) {
                            self.lazy.state.store(self.next_state, SeqCst);
                        }
                    }

                    {
                        let mut updater = Updater {
                            lazy: self,
                            next_state: LAZY_STATE_EMPTY,
                        };

                        let value = f();
                        // Now that f() has completed without panicking, we can update the
                        // next_state.
                        core::ptr::write((*self.cell.get()).as_mut_ptr(), value);
                        updater.next_state = LAZY_STATE_READY;
                    }
                    // At this point, the Drop handler has run, either during normal control flow
                    // or as a side-effect of unwinding the stack.
                    debug_assert_eq!(self.state.load(SeqCst), LAZY_STATE_READY);

                    return self.value_ref();
                },
                Err(LAZY_STATE_UPDATING) => {
                    // Another thread is updating the lock. Release the current CPU and try again.
                    // Spin until it is no longer UPDATING.
                    yield_thread();
                }
                Err(LAZY_STATE_READY) => {
                    // Another thread already finished updating the value; the value that we
                    // computed will not be used.
                    return unsafe { self.value_ref() };
                }
                Err(unrecognized_state) => {
                    panic!("Illegal state value {} in AtomicLazy", unrecognized_state)
                }
            }
        }
    }

    /// Attempts to set the value in the `AtomicLazy`.
    ///
    /// If this function succeeds in moving a value into the `AtomicLazy`, then it also returns
    /// a reference to the item.
    ///
    /// If this function fails, then `value` is discarded (dropped).
    pub fn try_set(&self, value: T) -> Result<&T, ()>
    where
        T: Send,
    {
        if let Ok(_) =
            self.state
                .compare_exchange(LAZY_STATE_EMPTY, LAZY_STATE_UPDATING, SeqCst, SeqCst)
        {
            unsafe {
                core::ptr::write((*self.cell.get()).as_mut_ptr(), value);
                self.state.store(LAZY_STATE_READY, SeqCst);
                Ok(&*(*self.cell.get()).as_ptr())
            }
        } else {
            Err(())
        }
    }

    /// Attempts to set the value in the `AtomicLazy`. If the value cannot be set, then it is
    /// returned to the caller. This is intended for use with data structures that are expensive
    /// to construct, or which have visible side-effects.
    ///
    /// If this function fails, then `value` is returned in the `Err`.
    ///
    /// ```
    /// # use atomic_lazy::AtomicLazy;
    ///
    /// static LAZY: AtomicLazy<Vec<i32>> = AtomicLazy::new();
    ///
    /// // First call succeeds
    /// assert!(LAZY.try_set_return(vec![1, 2, 3, 4, 5]).is_ok());
    /// assert_eq!(LAZY.try_set_return(vec![-10, -20, -30]), Err(vec![-10, -20, -30]));
    /// ```
    pub fn try_set_return(&self, value: T) -> Result<&T, T>
    where
        T: Send,
    {
        if let Ok(_) =
            self.state
                .compare_exchange(LAZY_STATE_EMPTY, LAZY_STATE_UPDATING, SeqCst, SeqCst)
        {
            unsafe {
                core::ptr::write((*self.cell.get()).as_mut_ptr(), value);
                self.state.store(LAZY_STATE_READY, SeqCst);
                Ok(&*(*self.cell.get()).as_ptr())
            }
        } else {
            Err(value)
        }
    }

    /// Allows for initializing large types "in-place".
    ///
    /// This method must be used with extreme care!
    ///
    /// This method attempts to take ownership of the `AtomicLazy` for updating. If it succeeds,
    /// then it returns `Ok`. *In this state, all other accesses to the `AtomicLazy` will block
    /// (spin) or fail*, so you *must* take great care not to stay in this state for a long time.
    ///
    /// After a thread has called `try_begin_init_unsafe`, it should immediately store a valid
    /// object by writing through the returned pointer. The pointer points into _uninitialized_
    /// memory, and the caller takes all responsibility for complying with the rules for
    /// uninitialized memory: Do not read uninitialized memory, and do not create a `&T` or `&mut T`
    /// that points into uninitialized memory.
    ///
    /// The caller of this function is also responsible for initializing the object, in-place.
    /// After that is complete, the caller *must* call `AtomicLazy::finish_init_unsafe`. This
    /// transitions the `AtomicLazy` to the "ready" state. After that, the caller should never
    /// use the raw pointer to the memory block again, and instead should use the `&T` returned
    /// from `AtomicLazy::finish_init_unsafe`.
    ///
    ///
    /// ```
    /// # use atomic_lazy::AtomicLazy;
    ///
    /// static LAZY_DATA: AtomicLazy<[u8; 4096]> = AtomicLazy::new();
    /// unsafe {
    ///     let ptr: *mut u8 = LAZY_DATA.try_begin_init_unsafe().unwrap().as_ptr() as *mut u8;
    ///     for i in 0..4096 {
    ///         core::ptr::write_volatile(ptr.offset(i), 0);
    ///     }
    ///     // Now all of the memory in LAZY_DATA is fully initialized. We can now safely access it
    ///     // through references. And at this point, we have not yet transitioned the AtomicLazy
    ///     // to "ready".
    ///     let slice: &mut [u8] = core::slice::from_raw_parts_mut(ptr, 4096);
    ///     slice[0..5].copy_from_slice(b"Hello");
    ///     // Now transition the AtomicLazy to the "ready" state.
    ///     LAZY_DATA.finish_init_unsafe();
    /// }
    /// ```
    pub unsafe fn try_begin_init_unsafe(&self) -> Result<NonNull<T>, ()>
    where
        T: Send,
    {
        match self
            .state
            .compare_exchange(LAZY_STATE_EMPTY, LAZY_STATE_UPDATING, SeqCst, SeqCst)
        {
            Ok(_) => Ok(NonNull::new_unchecked(self.value_mut_ptr())),
            Err(_) => Err(()),
        }
    }

    /// Completes the work of `try_begin_init_unsafe`.
    ///
    /// If a call to `try_begin_init_unsafe` succeeds, then the caller *must* also call
    /// `finish_init_unsafe`.
    pub unsafe fn finish_init_unsafe(&self) -> &T {
        match self
            .state
            .compare_exchange(LAZY_STATE_UPDATING, LAZY_STATE_EMPTY, SeqCst, SeqCst)
        {
            Ok(_) => self.value_ref(),
            Err(_) => panic!("Illegal state for call to finish_init_unsafe"),
        }
    }
}

impl<T: Clone + Send> Clone for AtomicLazy<T> {
    fn clone(&self) -> Self {
        match self.get() {
            Some(value) => Self {
                cell: UnsafeCell::new(MaybeUninit::new(value.clone())),
                state: AtomicU8::new(LAZY_STATE_READY),
            },
            None => Self::default(),
        }
    }
}

impl<T> Drop for AtomicLazy<T> {
    fn drop(&mut self) {
        match self.state.load(SeqCst) {
            LAZY_STATE_READY => unsafe {
                let value: *mut T = (*self.cell.get()).as_mut_ptr();
                core::ptr::drop_in_place(value);
            },
            LAZY_STATE_EMPTY => {
                // nothing to do
            }
            LAZY_STATE_UPDATING => panic!("AtomicLazy is in UPDATING state!"),
            unrecognized_state => {
                panic!("Illegal state value {} in AtomicLazy", unrecognized_state)
            }
        }
    }
}

#[cfg(feature = "std")]
impl<T> std::panic::RefUnwindSafe for AtomicLazy<T> {}

impl<T: Send> From<AtomicLazy<T>> for Option<T> {
    fn from(lazy: AtomicLazy<T>) -> Option<T> {
        // Because the value is being moved, we have exclusive access to it.
        match lazy.state.load(SeqCst) {
            LAZY_STATE_READY => unsafe {
                // Ensure that AtomicLazy::drop does not drop the inner value.
                // It is safe to store before reading the value out, because we know that there
                // are no references.
                lazy.state.store(LAZY_STATE_EMPTY, SeqCst);
                Some(core::ptr::read(lazy.value_mut_ptr()))
            },
            LAZY_STATE_EMPTY => None,
            LAZY_STATE_UPDATING => panic!("AtomicLazy is in UPDATING state!"),
            unrecognized_state => {
                panic!("Illegal state value {} in AtomicLazy", unrecognized_state)
            }
        }
    }
}

fn yield_thread() {
    unsafe {
        #[cfg(target_os = "windows")]
        SwitchToThread();

        #[cfg(target_os = "linux")]
        sched_yield();
    }
}

#[cfg(target_os = "windows")]
extern "system" {
    // https://docs.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-switchtothread
    fn SwitchToThread();
}

#[cfg(target_os = "linux")]
extern "C" {
    // https://linux.die.net/man/2/sched_yield
    fn sched_yield();
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Condvar, Mutex};

    #[test]
    fn test_empty() {
        let lazy: AtomicLazy<i32> = AtomicLazy::new();
        assert_eq!(lazy.get(), None);
    }

    #[test]
    fn test_try_set() {
        let lazy: AtomicLazy<i32> = AtomicLazy::new();
        assert_eq!(lazy.try_set(42), Ok(&42));
        assert_eq!(lazy.get(), Some(&42));
        assert_eq!(lazy.try_set(43), Err(()));
        assert_eq!(lazy.get(), Some(&42));
    }

    #[test]
    fn test_get_or_create() {
        let lazy: AtomicLazy<i32> = AtomicLazy::new();
        assert_eq!(lazy.get(), None);
        assert_eq!(lazy.get_or_create_with_dup(|| 42), &42);
        assert_eq!(lazy.get_or_create_with_dup(|| 43), &42);
    }

    struct StepSynchronizer {
        step: Mutex<u32>,
        condvar: Condvar,
    }

    impl StepSynchronizer {
        fn new() -> Arc<Self> {
            Arc::new(Self {
                step: Mutex::new(0),
                condvar: Condvar::new(),
            })
        }

        fn actor(self: &Arc<Self>, who: String) -> StepSynchronizerActor {
            StepSynchronizerActor {
                synchronizer: Arc::clone(self),
                who,
            }
        }
    }

    struct StepSynchronizerActor {
        synchronizer: Arc<StepSynchronizer>,
        who: String,
    }
    impl StepSynchronizerActor {
        fn step(&self, step: u32, description: &str) {
            println!(
                "step {}, thread {}, {} - waiting",
                step, self.who, description
            );
            let mut g = self.synchronizer.step.lock().unwrap();
            loop {
                assert!(*g <= step);
                if *g == step {
                    println!(
                        "step {}, thread {}, {} - done",
                        step, &self.who, description
                    );
                    *g += 1;
                    self.synchronizer.condvar.notify_one();
                    return;
                }
                g = self.synchronizer.condvar.wait(g).unwrap();
            }
        }
    }

    #[test]
    fn test_get_or_create_with_spin_success() {
        let lazy: AtomicLazy<i32> = AtomicLazy::new();
        assert_eq!(*lazy.get_or_create_with_spin(|| 100), 100);
    }

    #[test]
    fn test_get_or_create_with_spin_failure() {
        let lazy: AtomicLazy<i32> = AtomicLazy::new();
        assert_eq!(*lazy.get_or_create_with_spin(|| 100), 100);
        assert_eq!(*lazy.get_or_create_with_spin(|| 200), 100);
    }

    #[test]
    fn test_get_or_create_with_spin_panic() {
        let lazy: AtomicLazy<i32> = AtomicLazy::new();
        assert!(std::panic::catch_unwind(|| {
            lazy.get_or_create_with_spin(|| {
                panic!("Uh oh!");
            });
        })
        .is_err());
        assert_eq!(lazy.get(), None);
    }
    #[test]
    fn test_get_or_create_with_dup_panic() {
        let lazy: AtomicLazy<i32> = AtomicLazy::new();
        assert!(std::panic::catch_unwind(|| {
            lazy.get_or_create_with_dup(|| {
                panic!("Uh oh!");
            });
        })
        .is_err());
        assert_eq!(lazy.get(), None);
    }

    #[test]
    fn test_get_or_create_with_dup_success() {
        let lazy: AtomicLazy<i32> = AtomicLazy::new();
        assert_eq!(*lazy.get_or_create_with_dup(|| 100), 100);
    }

    #[test]
    fn test_get_or_create_with_dup_failure() {
        let lazy: AtomicLazy<i32> = AtomicLazy::new();
        assert_eq!(*lazy.get_or_create_with_dup(|| 100), 100);
        assert_eq!(*lazy.get_or_create_with_dup(|| 200), 100);
    }

    #[test]
    fn test_get_or_create_with_dup_contended() {
        let synchronizer = StepSynchronizer::new();
        let lazy: Arc<AtomicLazy<i32>> = Arc::new(AtomicLazy::new());

        let a_sync = synchronizer.actor("A".to_string());
        let a_lazy = Arc::clone(&lazy);

        let b_sync = synchronizer.actor("B".to_string());
        let b_lazy = Arc::clone(&lazy);

        let t = std::thread::spawn(move || {
            let this_thread = "B";
            let thread_2_result = b_lazy.get_or_create_with_dup(|| {
                b_sync.step(1, "in eval function");
                100
            });
            b_sync.step(3, "finished eval function, checking results");
            println!("results seen by {}: {}", this_thread, *thread_2_result);
            assert_eq!(*thread_2_result, 100);
            b_sync.step(5, "done");
        });

        // thread "B" should "win".
        {
            {
                let thread_1_result = a_lazy.get_or_create_with_dup(|| {
                    a_sync.step(0, "in eval function");
                    a_sync.step(2, "waiting for other thread in eval function");
                    200
                });
                a_sync.step(4, "finished eval function, checking results");
                assert_eq!(*thread_1_result, 100);
                a_sync.step(6, "stuff");
                println!("results seen by {}: {}", a_sync.who, *thread_1_result);
            }
            a_sync.step(7, "done");
        }

        println!("waiting for thread_2 to terminate");
        t.join().unwrap();

        println!("done");
    }

    #[test]
    fn test_drop_ready() {
        use std::sync::Arc;
        let cell = Arc::new(String::from("Hello, world"));
        assert_eq!(Arc::strong_count(&cell), 1);
        {
            let lazy: AtomicLazy<Arc<String>> = AtomicLazy::new();
            assert!(lazy.try_set(Arc::clone(&cell)).is_ok());
            assert_eq!(Arc::strong_count(&cell), 2);
        }
        assert_eq!(Arc::strong_count(&cell), 1);
    }

    #[test]
    fn test_drop_empty() {
        let lazy: AtomicLazy<String> = AtomicLazy::new();
        drop(lazy);
    }

    #[test]
    fn test_static() {
        static LAZY: AtomicLazy<String> = AtomicLazy::new();

        assert!(LAZY.get().is_none());
        assert!(LAZY.try_set(String::from("Hello, world")).is_ok());
        assert_eq!(LAZY.get(), Some(&String::from("Hello, world")));
    }

    #[test]
    fn test_into_none() {
        let lazy: AtomicLazy<String> = AtomicLazy::new();
        let result: Option<String> = lazy.into();
        assert_eq!(result, None);
    }

    #[test]
    fn test_into_some() {
        let lazy: AtomicLazy<String> = AtomicLazy::new();
        assert!(lazy.try_set(String::from("Hello, world")).is_ok());
        let result: Option<String> = lazy.into();
        assert_eq!(result, Some(String::from("Hello, world")));
    }
}
