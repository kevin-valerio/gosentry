package sigsegvrestart

/*
#include <signal.h>

static void trigger_sigsegv(void) {
	raise(SIGSEGV);
}
*/
import "C"

func triggerSIGSEGV() {
	C.trigger_sigsegv()
}
