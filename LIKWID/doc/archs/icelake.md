/*! \page skylake Intel&reg; Icelake

<P>This page is valid for Icelake. The Icelake microarchitecture supports the UBOX and the CBOX Uncore devices.</P>

<H1>Available performance monitors for the Intel&reg; Icelake microarchitecture</H1>
<UL>
<LI>\ref ICL_FIXED "Fixed-purpose counters"</LI>
<LI>\ref ICL_METRICS "Performance metric counters"</LI>
<LI>\ref ICL_PMC "General-purpose counters"</LI>
<LI>\ref ICL_THERMAL "Thermal counters"</LI>
<LI>\ref ICL_VOLTAGE "Core voltage counters"</LI>
<LI>\ref ICL_POWER "Power measurement counters"</LI>
<LI>\ref ICL_UBOX "Uncore global counters"</LI>
<LI>\ref ICL_CBOX "Last level cache counters"</LI>
<LI>\ref ICL_MBOX "Memory controller counters"</LI>
</UL>

<H1>Counters available for each hardware thread</H1>
\anchor ICL_FIXED
<H2>Fixed-purpose counters</H2>
<P>Since the Core2 microarchitecture, Intel&reg; provides a set of fixed-purpose counters. Each can measure only one specific event. The Intel&reg; Icelake architecture adds a fourth fixed-purpose counter for the event TOPDOWN_SLOTS.</P>
<H3>Counter and events</H3>
<TABLE>
<TR>
  <TH>Counter name</TH>
  <TH>Event name</TH>
</TR>
<TR>
  <TD>FIXC0</TD>
  <TD>INSTR_RETIRED_ANY</TD>
</TR>
<TR>
  <TD>FIXC1</TD>
  <TD>CPU_CLK_UNHALTED_CORE</TD>
</TR>
<TR>
  <TD>FIXC2</TD>
  <TD>CPU_CLK_UNHALTED_REF</TD>
</TR>
<TR>
  <TD>FIXC3</TD>
  <TD>TOPDOWN_SLOTS</TD>
</TR>
</TABLE>
<H3>Available Options</H3>
<TABLE>
<TR>
  <TH>Option</TH>
  <TH>Argument</TH>
  <TH>Description</TH>
  <TH>Comment</TH>
</TR>
<TR>
  <TD>anythread</TD>
  <TD>N</TD>
  <TD>Set bit 2+(index*4) in config register</TD>
  <TD></TD>
</TR>
<TR>
  <TD>kernel</TD>
  <TD>N</TD>
  <TD>Set bit (index*4) in config register</TD>
  <TD></TD>
</TR>
</TABLE>

\anchor ICL_METRICS
<H2>Performance metric counters</H2>
<P>With the Intel&reg; Icelake microarchitecture a new class of core-local counters was introduced, the so-called perf-metrics. The reflect the first level of the <A HREF="https://software.intel.com/content/www/us/en/develop/documentation/vtune-cookbook/top/methodologies/top-down-microarchitecture-analysis-method.html">Top-down Microarchitecture Analysis</A> tree. The events return the fraction of slots used by the event.</P>
<H3>Counter and events</H3>

<TABLE>
<TR>
  <TH>Counter name</TH>
  <TH>Event name</TH>
</TR>
<TR>
  <TD>TMA0</TD>
  <TD>RETIRING</TD>
</TR>
<TR>
  <TD>TMA1</TD>
  <TD>BAD_SPECULATION</TD>
</TR>
<TR>
  <TD>TMA2</TD>
  <TD>FRONTEND_BOUND</TD>
</TR>
<TR>
  <TD>TMA3</TD>
  <TD>BACKEND_BOUND</TD>
</TR>
</TABLE> 

\anchor ICL_PMC
<H2>General-purpose counters</H2>
<P>Commonly the Intel&reg; Icelake microarchitecture provides 4 general-purpose counters consisting of a config and a counter register.</P>
<H3>Counter and events</H3>
<TABLE>
<TR>
  <TH>Counter name</TH>
  <TH>Event name</TH>
</TR>
<TR>
  <TD>PMC0</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>PMC1</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>PMC2</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>PMC3</TD>
  <TD>*</TD>
</TR>
</TABLE>
<H3>Available Options</H3>
<TABLE>
<TR>
  <TH>Option</TH>
  <TH>Argument</TH>
  <TH>Description</TH>
  <TH>Comment</TH>
</TR>
<TR>
  <TD>edgedetect</TD>
  <TD>N</TD>
  <TD>Set bit 18 in config register</TD>
  <TD></TD>
</TR>
<TR>
  <TD>kernel</TD>
  <TD>N</TD>
  <TD>Set bit 17 in config register</TD>
  <TD></TD>
</TR>
<TR>
  <TD>anythread</TD>
  <TD>N</TD>
  <TD>Set bit 21 in config register</TD>
  <TD>The anythread option is deprecated but still supported. Please check the official Intel&reg; documentation how to use the option properly.</TD>
</TR>
<TR>
  <TD>threshold</TD>
  <TD>8 bit hex value</TD>
  <TD>Set bits 24-31 in config register</TD>
  <TD></TD>
</TR>
<TR>
  <TD>invert</TD>
  <TD>N</TD>
  <TD>Set bit 23 in config register</TD>
  <TD></TD>
</TR>
<TR>
  <TD>in_transaction</TD>
  <TD>N</TD>
  <TD>Set bit 32 in config register</TD>
  <TD>Only available if Intel&reg; Transactional Synchronization Extensions are available</TD>
</TR>
<TR>
  <TD>in_transaction_aborted</TD>
  <TD>N</TD>
  <TD>Set bit 33 in config register</TD>
  <TD>Only counter PMC2 and only if Intel&reg; Transactional Synchronization Extensions are available</TD>
</TR>
</TABLE>

<P>If HyperThreading is disabled, each core can use 8 general-purpose counters named &lt;4-7&gt;.


\anchor ICL_THERMAL
<H2>Thermal counters</H2>
<P>The Intel&reg; Icelake microarchitecture provides one register for the current core temperature.</P>
<H3>Counter and events</H3>
<TABLE>
<TR>
  <TH>Counter name</TH>
  <TH>Event name</TH>
</TR>
<TR>
  <TD>TMP0</TD>
  <TD>TEMP_CORE</TD>
</TR>
</TABLE>

\anchor ICL_VOLTAGE
<H2>Core voltage counters</H2>
<P>The Intel&reg; Icelake microarchitecture provides one register for the current core voltage.</P>
<H3>Counter and events</H3>
<TABLE>
<TR>
  <TH>Counter name</TH>
  <TH>Event name</TH>
</TR>
<TR>
  <TD>VTG0</TD>
  <TD>VOLTAGE_CORE</TD>
</TR>
</TABLE>

<H1>Counters available for one hardware thread per socket</H1>
\anchor ICL_POWER
<H2>Power counter</H2>
<P>The Intel&reg; Icelake microarchitecture provides measurements of the current power consumption through the RAPL interface.</P>
<H3>Counter and events</H3>
<TABLE>
<TR>
  <TH>Counter name</TH>
  <TH>Event name</TH>
</TR>
<TR>
  <TD>PWR0</TD>
  <TD>PWR_PKG_ENERGY</TD>
</TR>
<TR>
  <TD>PWR1</TD>
  <TD>PWR_PP0_ENERGY</TD>
</TR>
<TR>
  <TD>PWR2</TD>
  <TD>PWR_PP1_ENERGY</TD>
</TR>
<TR>
  <TD>PWR3</TD>
  <TD>PWR_DRAM_ENERGY</TD>
</TR>
<TR>
  <TD>PWR4</TD>
  <TD>PWR_PLATFORM_ENERGY</TD>
</TR>
</TABLE>

\anchor ICL_UBOX
<H2>Uncore global counters</H2>
<P>The Intel&reg; Icelake microarchitecture provides measurements for the global uncore.</P>
<H3>Counter and events</H3>
<TABLE>
<TR>
  <TH>Counter name</TH>
  <TH>Event name</TH>
</TR>
<TR>
  <TD>UBOX0</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>UBOX1</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>UBOXFIX</TD>
  <TD>UNCORE_CLOCK</TD>
</TR>
</TABLE>


\anchor ICL_CBOX
<H2>Last level cache counters</H2>
<P>The Intel&reg; Icelake microarchitecture provides measurements for the last level cache segments.</P>
<H3>Counter and events</H3>
<TABLE>
<TR>
  <TH>Counter name</TH>
  <TH>Event name</TH>
</TR>
<TR>
  <TD>CBOX&lt;0-3&gt;C0</TD>
  <TD>*</TD>
</TR>
<TR>
  <TD>CBOX&lt;0-3&gt;C1</TD>
  <TD>*</TD>
</TR>
</TABLE>

<H3>Available Options</H3>
<TABLE>
<TR>
  <TH>Option</TH>
  <TH>Argument</TH>
  <TH>Description</TH>
  <TH>Comment</TH>
</TR>
<TR>
  <TD>edgedetect</TD>
  <TD>N</TD>
  <TD>Set bit 18 in config register</TD>
  <TD></TD>
</TR>
<TR>
  <TD>threshold</TD>
  <TD>8 bit hex value</TD>
  <TD>Set bits 24-28 in config register</TD>
  <TD></TD>
</TR>
<TR>
  <TD>invert</TD>
  <TD>N</TD>
  <TD>Set bit 23 in config register</TD>
  <TD></TD>
</TR>
</TABLE>

\anchor ICL_MBOX
<H2>Memory controller counters</H2>
<P>The Intel&reg; Icelake microarchitecture provides measurements for the memory controller. The counters are free-running.</P>
<H3>Counter and events</H3>
<TABLE>
<TR>
  <TH>Counter name</TH>
  <TH>Event name</TH>
</TR>
<TR>
  <TD>MBOX0C0</TD>
  <TD>IO_REQUESTS</TD>
</TR>
<TR>
  <TD>MBOX0C1</TD>
  <TD>DRAM_READS</TD>
</TR>
<TR>
  <TD>MBOX0C2</TD>
  <TD>DRAM_WRITES</TD>
</TR>
</TABLE>
*/
