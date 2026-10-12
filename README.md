# Project Showcase for the MRes Project Allocation 

1. *Which parts of the Mercedes-Benz agent, its FastAPI backend and its deployment you built yourself, and what the company's software teams did?*

> The company's software team collaborated with me to give me access to the development branch and the deployment server/process and to let me know what tech stack my work would have to use to be integrated into their business platform. They also assigned to me the routes that my project would use in the overall application.
>
> I built the complete agent from scratch, starting from it's required tasks, building the execution graph, implementing the tools, wrapping it all in LangChain and LangGraph and then writing a public API on top of this agent to work with the backend. I then designed the backend myself, within the framework given to me by the senior engineer on the software team, and implemented this using FastAPI. This was basically just routing and wrapping my agent's API. I also wireframed, designed and implemented the entire front-end on VueJS, which was effectively a basic chat interface for LLMs, but it worked very differently due to the novelties introduced in my agent. This was why I had to do everything from scratch because existing LLM templates were not compatible (and heavily tended to be NextJS oriented!).
>
> In summary: my project was an independent undertaking and all functionality was implemented on my own in the 3 month duration of the internship, with close supervision from my direct manager. The software team collaborated with me in providing me the space to deploy my work and to speedrun the Mercedes tech-stack and programming standards with me to ensure the deployment was seamless.

---

2. *Which instructions your microprocessor implemented, as your CV describes it as a subset of RISC-V?*

> Base stats for the CPU: 8-bit data-bus and 8-bit memory addresses. The instructions and data are also 8-bits long. The processor has two registers 8-bit A and B that hold the operand values for most operations. The operations implemented are shown in the image below. 

<figure style="width: 80%; position: relative; inset: 0; margin: auto; border-radius: 10px; overflow: hidden; height: fit-content;">
    <image src="attachments/mpu-instructions.png" alt="image" style="border-radius: 10px;"/>
    <figcaption style="text-align: center">Table of all instructions implemented on the MPU that are also supported by the implemented assembler.</figcaption>
</figure>

<br/>

> This project was part of coursework undertaken during my undergraduate degree. This corresponds to the Digital Systems Laboratory course on my transcript from Edinburgh Uni (Result: 80/100). [This repository on GitHub](https://github.com/ArtConnoisseur/microprocessor-in-verilog/tree/main) contains the implementation for this project. There is no README for the repo, instead, as a part of the submission I provided README.pdfs which are linked here:

1. [Submission of the Microprocessor and the first version of the assembler - README.pdf](/microprocessor/README.pdf).
2. [Application built to drive the VGA and second version of the assembler - README.pdf](/microprocessor/README-II.pdf).

> To contextualise these files; the course worked by setting out the complete project specification from the beginning, and there were three total submissions at three sequential stages of implementation of the project. The README.pdf was not required but I decided to include one anyway. The Assembler also was **not part of this course**. I implemented this to speed up the assembly coding and iteration (it was either this or code in hexadecimal, the latter is not entirely fun nor conducive to iteration). I keep calling this an assembler, because that is what it functions as, but it's more so an embedded DSL for the ISA I implemented in my microprocessor. The files linked above explain this better.
>
> Please read the first file for a breakdown of the file structure in the code, and it details my contributions as it was an individual submission. The second file describes what my partner and I collaborated on. I am happy to detail my exact contributions to the overall project when we meet, but in general, I made the microprocessor and the VGA driver. Also, while it was not part of my expected work, but I did also contribute to the mouse driver, mainly in making the cursor information come in more smoothly to the frame buffer.

---

3. Which modules on your transcript correspond to "Embedded Programming", "Signals and Networking" and "Information Theory" on your CV, and whether Embedded Programming involved C or C++?


| CV Course | Corresponding Transcript Course | Additional Comments |
| -: | - | - | 
| Embedded Programming| Engineering Software  | Year 3 course where we programmed the Microblaze microprocessor on the Xilinx (AMD) Basys 3 board **using C**. |
| Signals and Networking | Signals and Communication Systems 3 | Year 3 course that taught the basics of signals worked, how communication devices worked and basic information theory. |
| Information Theory | Digital Communications  | Year 4 course that taught us information theory and data communication methods between devices. 
| | |

> I also programmed a Raspberry Pi for my thesis, this is the BEng Electronics and Electrical Engineering Project on my transcript. I've also done more advanced DSP in Digital Signal Analysis from the year 3 course I mentioned above, but I didn't think it was a good idea to crowd my CV. I think it's important to mention that these courses did not focus on networking in the way a CS course on networking would; happy to expand on this when we meet as well.

---

4. Your own part of the Triton kernel in Higher Order Attention, and whether the code is public?

> The code is not public. My collaborator and I pursued this project as a hobby. The kernel was just a modification of the kernel from the [paper](https://arxiv.org/pdf/2507.02754) I cited in my original email. The reason for this is because our idea, the modification we proposed, unfortunately did not entirely solve the problem we hoped to address. Further our results were not the best and we were unable to find the compute to iterate over enough experiments to actually make substantial progress, so we moved onto other ideas in AI we wanted to explore. 
> 
> It is difficult for me to pinpoint my exact contributions to the kernel, (a) because the majority of it is from the original Fast and Simplex paper, and, (b) we pair programmed the modifications at first, where I mainly ensured mathematical soundness of the calculations and output from the kernel. Eventually, when we figured the maths out, my collaborator continued on to finish the kernel, I spent my time figuring out the open source codebase of [Torchtitan](https://github.com/pytorch/torchtitan) to actually implement our idea, and interface it with the Triton kernels efficiently and correctly. [This is the kernel](/higher-order-attention/fwd_pass.py).
>
> Also in [`/kernels`](/kernels/), there are some CUDA kernels I've written myself that are closer to my skill level (i.e. not modified from another paper and written individually). Notably there is the [Softmax Attention](/kernels/softmaxAttn.cu) kernel, that I think is pretty indicative of where I am at with CUDA. I am more comfortable with CUDA than with Triton.

---

5. The most concurrent code you have written, and in which language. 

> I've majorly used Python's `threading` and `multiprocessing` when I need concurrency, probably as a consequence of the fact that Python is my most used language. I did quite a bit of this at Mercedes, which is a private codebase and I can't access that. We could discuss this in the meeting too. I did do some coursework when learning concurrency and I often use it for data preprocessing. I will add some of those files in [`/concurrency`](/concurrency/), but they're probably not the best response to "the most concurrent code I've written". 

---

*Final Question from your email broken down into points:*

Details on...

7. Whether your MRes research proposal on mechanistic interpretability is still an option for you?

> [This proposal](/attachments/mres-mechinterp-proposal.pdf) is something I care deeply about and would love to explore this direction with you when we speak. However, from what I've been told this proposal was only supposed to be hypothetical to assess research taste and the ability to draft and plan a research problem as a part of the application materials. Additionally, given the extremely fast pace of mech-interp research and the more recent papers in this space, I imagine this proposal would have to be substantially modified. 

8. Whether you are also discussing projects with other supervisors? 

> Yes I am currently discussing projects with other supervisors.

9. Whether you plan a PhD in this area?

> I'm not sure if this refers to the projects I've chosen or Mechanistic Interpretability, therefore I will answer generally and hope that I answer your question. I fully intend on doing a PhD after my MRes. I hope that the PhD will be with my MRes supervisor and I am not selecting projects in consideration for only the MRes, but also based on topics I am interested to do my PhD in. I think I can answer your question more clearly in-person, so I look forward to discussing this in person. 